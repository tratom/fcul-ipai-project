#!/usr/bin/env python
# coding: utf-8

# ## Project Phase 1 - Aviation Accident Data Integration
# ### Group 03:
# - Tommaso Tragno - fc64699
# - Manuel Cardoso - fc56274
# - Chen Cheng - fc64872
# - Cristian Tedesco - fc65149

# #### Setup

import pandas as pd
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import os
import time
import seaborn as sns
import calendar
import py_stringmatching as sm
import re
from tqdm import tqdm
from pathlib import Path

PATH = 'data_sources/'
FILTERED_PATH = 'filtered/'

NTSB_DATA = 'ntsb-us-2003-2023.json'
AIR_TRAFFIC_DATA = 'u-s-airline-traffic-data.csv'
AIRCRAFT_DATA = 'aircraft_data.csv' #'aircraft_data_cleaned.csv' # the "cleaned" one contains the data cleaning part
WEATHER_DATA = 'weather_results.json'


# #### Load NTSB JSON Data

with open(PATH+NTSB_DATA, 'r', encoding='utf-8') as f:
    ntsb_raw_data = json.load(f)

# Each record is one accident/incident entry in a list
print(f'\n--- NTSB JSON loaded: {len(ntsb_raw_data)} total records found ---')

# Convert to a DataFrame (this will flatten top-level fields)
# For nested fields like 'Vehicles', we might do a separate flatten later
df_ntsb = pd.json_normalize(ntsb_raw_data, 
                            meta=[
                                'Oid','MKey','Closed','CompletionStatus','HasSafetyRec',
                                'HighestInjury','IsStudy','Mode','NtsbNumber',
                                'OriginalPublishedDate','MostRecentReportType','ProbableCause',
                                'City','Country','EventDate','State','Agency','BoardLaunch',
                                'BoardMeetingDate','DocketDate','EventType','Launch','ReportDate',
                                'ReportNum','ReportType','AirportId','AirportName','AnalysisNarrative',
                                'FactualNarrative','PrelimNarrative','FatalInjuryCount','MinorInjuryCount',
                                'SeriousInjuryCount','InvestigationClass','AccidentSiteCondition',
                                'Latitude','Longitude','DocketOriginalPublishDate'
                            ],
                            record_path=['Vehicles'],  # This flattens out the 'Vehicles' array
                            record_prefix='Vehicles.'
                           )

print('\n--- Flattened NTSB DataFrame (including Vehicles info): ---')

# print(df_ntsb.info())

# combines all injury counts to 1 column
df_ntsb['TotalInjuryCount'] = df_ntsb[['FatalInjuryCount', 'MinorInjuryCount', 'SeriousInjuryCount']].sum(axis=1)

# dropping unnecessary columns
df_ntsb.drop(columns=['AnalysisNarrative','FactualNarrative','PrelimNarrative','InvestigationClass','BoardLaunch'
                      ,'BoardMeetingDate','Launch','IsStudy','OriginalPublishedDate','DocketOriginalPublishDate'
                      ,'ReportType','ReportNum','ReportDate','MostRecentReportType','FatalInjuryCount','MinorInjuryCount'
                      ,'SeriousInjuryCount','DocketDate','Mode','HasSafetyRec','CompletionStatus','Closed'
                      ,'Vehicles.AircraftCategory','Vehicles.AmateurBuilt','Vehicles.EventID','Vehicles.AirMedical'
                      ,'Vehicles.AirMedicalType','Vehicles.flightScheduledType','Vehicles.flightServiceType'
                      ,'Vehicles.flightTerminalType','Vehicles.RegisteredOwner','Vehicles.RegulationFlightConductedUnder'
                      ,'Vehicles.RepGenFlag','Vehicles.RevenueSightseeing','Vehicles.SecondPilotPresent','Vehicles.Damage'
                      ,'AccidentSiteCondition'], inplace=True) 

# dropping NaT entries from EventDate
df_ntsb = df_ntsb.dropna(subset=['EventDate'])

# Type Conversion
df_ntsb['EventDate'] = pd.to_datetime(df_ntsb['EventDate']).dt.tz_localize(None)
df_ntsb['Vehicles.VehicleNumber'] = pd.to_numeric(df_ntsb['Vehicles.VehicleNumber'], errors='coerce').astype(int)
df_ntsb['MKey'] = pd.to_numeric(df_ntsb['MKey'], errors='coerce').astype(int)
df_ntsb['Vehicles.NumberOfEngines'] = pd.to_numeric(df_ntsb['Vehicles.NumberOfEngines'], errors='coerce').fillna(0).astype(int)
df_ntsb['Latitude'] = pd.to_numeric(df_ntsb['Latitude'], errors='coerce').astype(float)
df_ntsb['Longitude'] = pd.to_numeric(df_ntsb['Longitude'], errors='coerce').astype(float)
df_ntsb['TotalInjuryCount'] = pd.to_numeric(df_ntsb['TotalInjuryCount'], errors='coerce').astype(int)

categorical_cols = [
    'Vehicles.DamageLevel',
    'Vehicles.ExplosionType',
    'Vehicles.FireType',
    'HighestInjury',
    'EventType',
    'AccidentSiteCondition'
]

for col in categorical_cols:
    if col in df_ntsb.columns:
        df_ntsb[col] = df_ntsb[col].astype('category')

df_ntsb = df_ntsb.map(lambda x: x.lower() if isinstance(x, str) else x) # make all appropriate values lowercase

print(df_ntsb.info())

# #### Load Weather JSON Data
# (after fetching the data from open-meteo API)


with open(PATH+WEATHER_DATA, 'r', encoding='utf-8') as f:
    weather_raw_data = json.load(f)

# Each record is one day weather entry in a list
print(f'\n--- Weather JSON loaded: {len(weather_raw_data)} total records found ---')

# weather_data is a dict, e.g.:
# {
#   "cen24la079_2023-12-31_41.610278_-90.588361": {
#       "time": [...],
#       "temperature_2m": [...],
#       ...
#   }
# }

# Flatten into a tabular structure
all_rows = []
num_skip = 0

for accident_id, subdict in weather_raw_data.items():
    # subdict is a dict with keys like "time", "temperature_2m", ...
    # Each key is an array of the same length (24 hours).
    times = subdict.get("time", None)
    if times is None:
        print(f'Skipping {accident_id}: no "time" found.')
        num_skip += 1
        continue
    num_hours = len(subdict["time"])
    for i in range(num_hours):
        row = {"AccidentID": accident_id}  # store the top-level key
        for param, values_array in subdict.items():
            # param: "time", "temperature_2m", ...
            row[param] = values_array[i]  # pick the ith hour’s value
        all_rows.append(row)

df_weather = pd.DataFrame(all_rows)

# The missing values exists because not all accident have position data
# this cause the api to return empty data.
print("Skipped {} records over {} accidents.".format(num_skip, len(weather_raw_data.items())))

# Type conversion
df_weather["time"] = pd.to_datetime(df_weather["time"], errors="coerce")

int_columns = [
    "relative_humidity_2m",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_direction_10m",
    "wind_direction_100m",
    "weather_code"
]
float_columns = [
    "temperature_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_gusts_10m",
    "snow_depth"
]
for col in int_columns:
    df_weather[col] = pd.to_numeric(df_weather[col], errors="coerce").astype(int)
for col in float_columns:
    df_weather[col] = pd.to_numeric(df_weather[col], errors="coerce").astype(float)


print("\n--- Weather DataFrame sample ---")
print(df_weather.info())

# #### Load Airline Traffic CSV Data

df_airline_traffic = pd.read_csv(PATH+AIR_TRAFFIC_DATA, encoding='utf-8')

print(f'\n--- Airline CSV loaded: {df_airline_traffic.shape[0]} rows, {df_airline_traffic.shape[1]} columns ---')

# dropping unnecessary columns
df_airline_traffic.drop(columns=['Dom_RPM','Int_RPM','RPM','Dom_ASM','Int_ASM','ASM'], inplace=True) 

# print(df_airline_traffic.info())

# Remove commas from all columns and then convert
df_airline_traffic = df_airline_traffic.replace(',', '', regex=True)

# Now convert each column to numeric. If everything converts well, no rows become NaN.
df_airline_traffic = df_airline_traffic.apply(pd.to_numeric, errors='coerce').astype(int)

print(df_airline_traffic.info())

# #### Load Aircraft CSV Data

df_aircraft = pd.read_csv(PATH+AIRCRAFT_DATA, encoding='utf-8')

print(f'\n--- Aircraft CSV loaded: {df_aircraft.shape[0]} rows, {df_aircraft.shape[1]} columns ---')

# print(df_aircraft.info())

# dropping unnecessary columns
df_aircraft.drop(columns=['Unnamed: 0'], inplace=True)
df_aircraft.drop(columns=['retired'], inplace=True)

# make string values lowercase
df_aircraft['aircraft'] = df_aircraft['aircraft'].str.lower()

# Type Conversion
df_aircraft['nbBuilt'] = pd.to_numeric(df_aircraft['nbBuilt'], errors='coerce').astype(int)
df_aircraft['startDate'] = pd.to_numeric(df_aircraft['startDate'], errors='coerce').astype(int)
df_aircraft['endDate'] = pd.to_numeric(df_aircraft['endDate'], errors='coerce').astype('Int64')  # Use 'Int64' for nullable integers

print(df_aircraft.info())

# ### 2. Data Profiling

def profile_dataframe(df, name='DataFrame'):
    print(f'\n=== Profiling {name} ===')
    print(f'Total Rows: {len(df)}')
    print(f'Total Columns: {len(df.columns)}\n')

    profile_results = []

    for col in df.columns:
        series = df[col]
        col_dtype = series.dtype

        # Basic counts
        total_count = len(series)
        missing_vals = series.isna().sum()
        non_null_count = total_count - missing_vals
        missing_perc = (missing_vals / total_count) * 100
        unique_vals = series.nunique(dropna=False)

        # Mode & frequency
        try:
            modes = series.mode(dropna=True)
            mode_val = modes.iloc[0] if len(modes) > 0 else np.nan
            mode_freq = (series == mode_val).sum(skipna=True)
        except:
            mode_val, mode_freq = np.nan, np.nan

        # Initialize placeholders
        mean_ = np.nan
        min_  = np.nan
        q25   = np.nan
        q50   = np.nan
        q75   = np.nan
        max_  = np.nan
        std_  = np.nan  # only for numeric columns

        # Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            mean_ = series.mean(skipna=True)
            min_  = series.min(skipna=True)
            q25   = series.quantile(0.25)
            q50   = series.quantile(0.50)
            q75   = series.quantile(0.75)
            max_  = series.max(skipna=True)
            std_  = series.std(skipna=True)

        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(series):
            # We can compute mean & quartiles by time. 
            # .quantile() and .mean() are valid for datetime in pandas
            # They return a Timestamp for mean, 
            # and Timestamps for quantiles
            if non_null_count > 0:
                mean_ = series.mean(skipna=True)
                min_  = series.min(skipna=True)
                q25   = series.quantile(0.25)
                q50   = series.quantile(0.50)
                q75   = series.quantile(0.75)
                max_  = series.max(skipna=True)
            # We skip std_ for datetime.

        # Categorical/object columns 
        # do not get numeric stats (we keep them as NaN).

        profile_results.append((
            col,
            str(col_dtype),
            total_count,
            non_null_count,
            missing_vals,
            round(missing_perc, 2),
            unique_vals,
            mode_val,
            mode_freq,
            mean_,
            min_,
            q25,
            q50,
            q75,
            max_,
            std_
        ))

    columns = [
        'Column', 'DataType', 'TotalCount', 'NonNullCount', 'NumMissing',
        'MissingPerc', 'Cardinality', 'Mode', 'ModeFreq',
        'Mean', 'Min', 'Q25', 'Q50', 'Q75', 'Max', 'Std'
    ]

    prof_df = pd.DataFrame(profile_results, columns=columns)

    return prof_df

# #### NTSB Data Profile

ntsb_profile = profile_dataframe(df_ntsb, name='NTSB Data')
display(HTML(ntsb_profile.to_html()))
ntsb_profile.to_csv(PATH+'profiling/ntsb_profile.csv', index=False)


# Insights from the data profile results:
# 
# - there are some `null` values for Latitude and Longitude --> we keep like this, but they should be handled during the API calls to open-meteo
# - there are less unique `NtsbNumber` than rows --> for incident where more than one aircraft is involved, the rows are duplicated with different values for Vehicles characteristic, and same value for incident data (look at the following example)

df_ntsb.loc[df_ntsb['NtsbNumber']=='ops24la011']


# #### Weather Data Profile

weather_profile = profile_dataframe(df_weather, name='Weather Data')
display(HTML(weather_profile.to_html()))
weather_profile.to_csv(PATH+'profiling/weather_profile.csv', index=False)


# #### Air Traffic Data Profile

airline_profile = profile_dataframe(df_airline_traffic, name='Airline Data')
display(HTML(airline_profile.to_html()))
airline_profile.to_csv(PATH+'profiling/airline_profile.csv', index=False)


# #### Aircraft Data Profile

aircraft_profile = profile_dataframe(df_aircraft, name='Aircraft Data')
display(HTML(aircraft_profile.to_html()))
aircraft_profile.to_csv(PATH+'profiling/aircraft_profile.csv', index=False)

# Insights from the data profile results:
# 
# - there are some `startDate` and `endDate` equal to 1 --> it is supposed to be a year

df_filtered = df_aircraft[(df_aircraft['startDate'] < 1000) | (df_aircraft['endDate'] < 1000)]
df_filtered.style.map(
    lambda val: 'background-color: red' if val < 1000 else '',
    subset=['startDate', 'endDate']
)


# ### Charts

# Group by 'Month' and sum 'Flt'
monthly_flt_sum = df_airline_traffic.groupby('Month')['Flt'].sum().reset_index()

# Sort by month to be sure
monthly_flt_sum = monthly_flt_sum.sort_values('Month')

# Map month numbers to names (Jan, Feb, ...)
month_names = [calendar.month_abbr[m] for m in monthly_flt_sum['Month']]
monthly_flt_sum['Month_Name'] = month_names

# Display result
print(monthly_flt_sum)

# Histogram
# Plot
plt.figure(figsize=(10, 6))
plt.bar(monthly_flt_sum['Month_Name'], monthly_flt_sum['Flt'], color='skyblue', edgecolor='black')

# Labels and title
plt.title('Total Flights per Month (All Years)')
plt.xlabel('Month')
plt.ylabel('Total Flights')

plt.tight_layout()
plt.show()

# Box Plot
# Map numeric month to abbreviation
df_airline_traffic['Month_Name'] = df_airline_traffic['Month'].apply(lambda x: calendar.month_abbr[x])

# Optional: Order months correctly
month_order = list(calendar.month_abbr)[1:]  # ['Jan', 'Feb', ..., 'Dec']

# Plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_airline_traffic, x='Month_Name', y='Flt', order=month_order, palette='pastel')

# Labels and title
plt.title('Distribution of Flights per Month (All Years)')
plt.xlabel('Month')
plt.ylabel('Number of Flights')

plt.tight_layout()
plt.show()


# ## Blocking strategy

# === 1. Data Loading ===
# Caricamento dei dataset
df_aircraft = pd.read_csv('data_sources/combined_aircraft_data.csv')

# Selezione delle colonne necessarie
df_ntsb_model = df_ntsb[['NtsbNumber', 'EventDate', 'Vehicles.SerialNumber',
                         'Vehicles.RegistrationNumber', 'Vehicles.Make', 'Vehicles.Model']].copy()

# === 2. Data Cleaning and Normalization ===
def clean_text(s):
    """ Normalizzazione del testo: rimozione di caratteri speciali, lowercase e spazi extra. """
    return re.sub(r'\W+', ' ', str(s)).lower().strip()

# Pulizia dei dati
df_ntsb_model['Vehicles.Model'] = df_ntsb_model['Vehicles.Model'].apply(clean_text)
df_ntsb_model['Vehicles.Make'] = df_ntsb_model['Vehicles.Make'].apply(clean_text)

df_aircraft['model_no'] = df_aircraft['model_no'].apply(clean_text)
df_aircraft['manufacturer_code'] = df_aircraft['manufacturer_code'].apply(clean_text)

df_ntsb_model.dropna(subset=['Vehicles.Model'], inplace=True)
df_aircraft.dropna(subset=['model_no'], inplace=True)

# === 3. Similarity Setup ===
jw = sm.JaroWinkler()
lev = sm.Levenshtein()
jac = sm.Jaccard()  

# === 4. Precomputation degli n-gram ===
def generate_qgrams(model):
    """ Genera un insieme di trigrammi (q-grams di lunghezza 3) per una stringa data. """
    qgrams = [model[i:i+3] for i in range(len(model) - 2)]
    return set(qgrams)

# === 5. Matching with Optimized Loop ===
matches = []
matched_set = set()  # Set per controllare i duplicati di NtsbNumber + SerialNumber
serial_set = set()   # Set per controllare i duplicati di SerialNumber

for i, ntsb_row in df_ntsb_model.iterrows():
    model_ntsb = ntsb_row['Vehicles.Model']
    make_ntsb = ntsb_row['Vehicles.Make']
    grams_ntsb = generate_qgrams(model_ntsb)

    # 🔎 **Filtro preliminare basato sul Make (flessibile)**
    filtered_aircraft = df_aircraft[
        df_aircraft['manufacturer_code'].apply(lambda x: make_ntsb in x or x in make_ntsb or jw.get_sim_score(x, make_ntsb) > 0.85)
    ]

    # Se non ci sono candidati, passa al prossimo ciclo
    if filtered_aircraft.empty:
        continue

    # Precomputa gli n-gram per i candidati
    aircraft_grams = {index: generate_qgrams(model) for index, model in enumerate(filtered_aircraft['model_no'])}

    # 🔎 **Filtro preliminare basato sugli n-grammi**
    candidate_matches = []
    for idx, grams_aircraft in aircraft_grams.items():
        if len(grams_ntsb & grams_aircraft) >= 2:
            candidate_matches.append(filtered_aircraft.index[idx])

    if not candidate_matches:
        continue  # Nessun match possibile, passo al successivo

    # 🔎 **Controllo diretto:** se esiste un match esatto tra i candidati
    direct_match = df_aircraft.loc[candidate_matches]
    direct_match = direct_match[direct_match['model_no'] == model_ntsb]

    if not direct_match.empty:
        for _, row in direct_match.iterrows():
            match_id = f"{ntsb_row['NtsbNumber']}_{ntsb_row['Vehicles.SerialNumber']}_{row['model_no']}"
            if match_id not in matched_set and ntsb_row['Vehicles.SerialNumber'] not in serial_set:
                matches.append({
                    'NtsbNumber': ntsb_row['NtsbNumber'],
                    'EventDate': ntsb_row['EventDate'],
                    'Vehicles.SerialNumber': ntsb_row['Vehicles.SerialNumber'],
                    'Vehicles.RegistrationNumber': ntsb_row['Vehicles.RegistrationNumber'],
                    'Vehicles.Make': ntsb_row['Vehicles.Make'],
                    'Vehicles.Model': model_ntsb,
                    'Matched_Aircraft_Model': row['model_no'],
                    'engine_count': row['engine_count'],
                    'engine_type': row['engine_type'],
                    'JW_Score': 1.0,
                    'LEV_Score': 1.0,
                    'Jac_Score': 1.0,
                    'SimilarityScore': 1.0
                })
                matched_set.add(match_id)
                serial_set.add(ntsb_row['Vehicles.SerialNumber'])

        continue  # Salta il loop di matching

    # 🔎 **Controllo di Variante Generico**
    for idx in candidate_matches:
        model_aircraft = df_aircraft.loc[idx, 'model_no']

        # Numeric Filter: numbers must match if present
        nums_ntsb = re.findall(r'\d+', model_ntsb)
        nums_aircraft = re.findall(r'\d+', model_aircraft)

        if nums_ntsb and nums_aircraft and nums_ntsb != nums_aircraft:
            continue

        # Computing the Three Similarity Scores
        jw_score = jw.get_sim_score(model_ntsb, model_aircraft)
        lev_score = lev.get_sim_score(model_ntsb, model_aircraft)
        jac_score = jac.get_sim_score(list(grams_ntsb), list(generate_qgrams(model_aircraft)))

        # Linear Rule
        final_score = 0.4 * jw_score + 0.3 * lev_score + 0.3 * jac_score

        # Controllo duplicati
        match_id = f"{ntsb_row['NtsbNumber']}_{ntsb_row['Vehicles.SerialNumber']}_{model_aircraft}"
        if final_score > 0.75 and match_id not in matched_set and ntsb_row['Vehicles.SerialNumber'] not in serial_set:
            matches.append({
                'NtsbNumber': ntsb_row['NtsbNumber'],
                'EventDate': ntsb_row['EventDate'],
                'Vehicles.SerialNumber': ntsb_row['Vehicles.SerialNumber'],
                'Vehicles.RegistrationNumber': ntsb_row['Vehicles.RegistrationNumber'],
                'Vehicles.Make': ntsb_row['Vehicles.Make'],
                'Vehicles.Model': model_ntsb,
                'Matched_Aircraft_Model': model_aircraft,
                'engine_count': df_aircraft.loc[idx, 'engine_count'],
                'engine_type': df_aircraft.loc[idx, 'engine_type'],
                'JW_Score': round(jw_score, 3),
                'LEV_Score': round(lev_score, 3),
                'Jac_Score': round(jac_score, 3),
                'SimilarityScore': round(final_score, 4)
            })
            matched_set.add(match_id)
            serial_set.add(ntsb_row['Vehicles.SerialNumber'])

# === 6. Final Output ===
if not matches:
    print("No matches found with the current rules.")
else:
    df_matches = pd.DataFrame(matches)
    print(f"Matches Found: {len(df_matches)}")
    print("Columns:", df_matches.columns.tolist())
    df_matches = df_matches.sort_values(by='SimilarityScore', ascending=False)

display(df_matches)

# # Data Fusion Strategy

ntsb_copied = df_ntsb.copy()
ntsb_copied = ntsb_copied.rename(columns=
    {"EventDate": "Date", 
    "NtsbNumber": "ID", 
    "State": "Location"}) # for schema matching 

seed = 10
np.random.seed(seed)

n = len(ntsb_copied)

# select 60% indices for later use
n_forty = int(np.floor(0.6 * n))
random_indices = np.random.choice(ntsb_copied.index, n_forty, replace=False)
print(f"random_indices count (60%): {len(random_indices)}")  # Should be ~0.6 * n

# select half of these indices to assign NaN in 'State'
n_missing = int(np.floor(0.5 * n_forty))
missing_indices = np.random.choice(random_indices, n_missing, replace=False)

# assign NaN only to these missing_indices, in original to then match
df_ntsb.loc[missing_indices, "State"] = np.nan  # for slot filling

print(f"Number of NaNs assigned: {len(missing_indices)}")

# count total NaNs in 'ID' (including existing NaNs)
total_nans = df_ntsb['State'].isna().sum()
print(f"Total NaNs in 'State' column: {total_nans}")

conflict_indices = np.setdiff1d(random_indices, missing_indices)

# transform strings for conflict resolution
for index, row in ntsb_copied.iterrows():
    if index in conflict_indices:
        airport = row["AirportName"]
        if pd.notna(airport):
            result = ' '.join([word[0] + '.' for word in airport.split()])
            ntsb_copied.loc[index, "AirportName"] = result

# ### Fuse NTSB with its dupe for strategy implemenation

tqdm.pandas()

# Load datasets

# NaN count BEFORE fusion
id_nans_before = df_ntsb['State'].isna().sum()
print(f"NaN in 'State' (before fusion): {id_nans_before}")

# Count abbreviated AirportName BEFORE fusion (optional)
abbrev_pattern = r'^([a-zA-Z]\.\s*)+$'
airport_abbrev_before = ntsb_copied['AirportName'].fillna('').str.match(abbrev_pattern).sum()
print(f"Abbreviated AirportName (before fusion): {airport_abbrev_before}")

# Normalize column names in ntsb_copied
ntsb_copied = ntsb_copied.rename(columns={
    "Date": "EventDate", 
    "ID": "NtsbNumber", 
    "Location": "State"
})

# Drop NtsbNumber
df_ntsb = df_ntsb.drop(columns=['NtsbNumber'], errors='ignore')
ntsb_copied = ntsb_copied.drop(columns=['NtsbNumber'], errors='ignore')

# Merge logic
fused_rows = []
unmatched_rows = []

for _, row in tqdm(ntsb_copied.iterrows(), total=len(ntsb_copied), desc="Merging entries"):
    match = df_ntsb[
        (df_ntsb['EventDate'] == row['EventDate']) &
        (df_ntsb['Vehicles.RegistrationNumber'] == row['Vehicles.RegistrationNumber']) &
        (df_ntsb['Vehicles.SerialNumber'] == row['Vehicles.SerialNumber'])
    ]

    if match.empty:
        unmatched_rows.append(row)
        continue

    merged = match.iloc[0].copy()

    # Slot Filling: if merged[col] is NA and row[col] is not, use row[col]
    for col in ntsb_copied.columns:
        if col in merged.index and pd.isna(merged[col]) and pd.notna(row[col]):
            merged[col] = row[col]

    # Conflict resolution: prioritize df unless row[col] is clearly more complete
    if pd.notna(row['AirportName']) and pd.notna(merged['AirportName']):
        if len(row['AirportName']) > len(merged['AirportName']):  # assume longer name is better
            merged['AirportName'] = row['AirportName']

    fused_rows.append(merged)

# Construct final fused dataframe
fused_df = pd.DataFrame(fused_rows)

# Reattach unmatched rows (optional)
fused_df = pd.concat([fused_df, pd.DataFrame(unmatched_rows)], ignore_index=True)

before_dedup = len(fused_df)
# Deduplicate
fused_df = fused_df.drop_duplicates(subset=[
    'EventDate', 'Vehicles.SerialNumber', 'Vehicles.RegistrationNumber'
])

# Save final fused result
fused_df.to_pickle('data_sources/fused/ntsb_fused.pkl')
fused_df.to_csv('data_sources/fused/ntsb_fused.csv', index=False)

print("Final fused dataset saved:")
print(" • accident_weather_fused_final.pkl")
print(" • accident_weather_fused_final.csv")
print(f"Final row count: {len(fused_df)} (original: {len(df_ntsb)}, added: {len(unmatched_rows)})")

# NaN count AFTER fusion
id_nans_after = fused_df['State'].isna().sum()
print(f"NaN in 'State' (after fusion): {id_nans_after}")
filled_count = id_nans_before - id_nans_after
nan_percent = filled_count / id_nans_before * 100
print(f"'State' values filled during fusion: {filled_count} ({nan_percent:.2f}%)")

# Deduplication stats
after_dedup = len(fused_df)
dedup_removed = before_dedup - after_dedup
dedup_percent = (dedup_removed / before_dedup) * 100
print(f"Deduplication removed: {dedup_removed} rows ({dedup_percent:.2f}%)")

# Count abbreviated AirportName AFTER fusion
abbrev_pattern = r'^([a-zA-Z]\.\s*)+$'
airport_abbrev_after = fused_df['AirportName'].fillna('').str.match(abbrev_pattern).sum()
print(f"Abbreviated AirportName (after fusion): {airport_abbrev_after}")
resolved_count = airport_abbrev_before - airport_abbrev_after
resolved_percent = resolved_count / airport_abbrev_before * 100
print(f"Resolved AirportName values: {resolved_count} ({resolved_percent:.2f}%)")


# ## Weather Data Fusion

# spatial & temporal thresholds
LAT_LON_EPS   = 0.10       # ≈ 11 km at mid-latitudes
MAX_TIME_DIFF = pd.Timedelta('3h')   # reject candidates > 3 h away

# --- 1. Load data -------------------------------------------------------------

ntsb    = df_ntsb
weather = df_weather

# ensure correct dtypes
ntsb["EventDate"] = pd.to_datetime(ntsb["EventDate"], errors="coerce")
weather["time"]   = pd.to_datetime(weather["time"],   errors="coerce")

# --- 2. Blocking on event *date* ---------------------------------------------
ntsb["event_day"]    = ntsb["EventDate"].dt.date
weather["weather_day"] = weather["time"].dt.date

weather_by_day = {d: w.reset_index(drop=True)
                  for d, w in weather.groupby("weather_day")}

# --- 3. Similarity matching & temporal precedence -----------------------------
best_rows = []       # stores best-matching weather rows (or None)

for _, acc in ntsb.iterrows():
    day_candidates = weather_by_day.get(acc["event_day"], pd.DataFrame())
    if day_candidates.empty:
        best_rows.append(None); continue

    # coarse spatial filter  |lat/lon diff| < LAT_LON_EPS
    spatial = day_candidates[
        (day_candidates["time"].notna()) &
        (day_candidates["AccidentID"].notna()) &       # keeps malformed rows out
        (day_candidates["AccidentID"].str.contains('_'))  # quick sanity
    ].copy()

    spatial = spatial[
        (np.abs(spatial["AccidentID"].str.split('_').str[-2].astype(float) - acc["Latitude" ] ) < LAT_LON_EPS) &
        (np.abs(spatial["AccidentID"].str.split('_').str[-1].astype(float) - acc["Longitude"]) < LAT_LON_EPS)
    ]

    if spatial.empty:
        best_rows.append(None); continue

    # temporal distance to the accident moment
    spatial["time_diff"] = (spatial["time"] - acc["EventDate"]).abs()

    # keep the closest hour that is still within MAX_TIME_DIFF
    spatial = spatial[spatial["time_diff"] <= MAX_TIME_DIFF]

    best_rows.append(spatial.nsmallest(1, "time_diff").iloc[0] if not spatial.empty else None)

# --- 4. Assemble the fused dataset -------------------------------------------
weather_match_df = pd.DataFrame.from_records(
    [row.to_dict() if row is not None else {}          # convert None into an empty dict {}
     for row in best_rows],
    index=ntsb.index                                   # keeps row-alignment
)

accident_weather = pd.concat(
    [ntsb.reset_index(drop=True),
     weather_match_df.add_prefix("wx_")],              # prefix to avoid clashes
    axis=1
)

# --- 5. Quick diagnostics -----------------------------------------------------
total_accidents = len(ntsb)
matched         = accident_weather["wx_time"].notna().sum()
print(f"Matched {matched} of {total_accidents} accidents "
      f"({matched / total_accidents:.1%})")

if matched:
    print("\nTime difference (min) for matched rows:")
    print((accident_weather.loc[accident_weather.wx_time.notna(), "wx_time_diff"]
           .dt.total_seconds().div(60)
           .describe().round(2)))

    print("\nSpatial deltas (deg lat/lon) for matched rows:")
    lat_delta = np.abs(accident_weather["Latitude"] - accident_weather["wx_AccidentID"]
                       .str.split('_').str[-2].astype(float))
    lon_delta = np.abs(accident_weather["Longitude"] - accident_weather["wx_AccidentID"]
                       .str.split('_').str[-1].astype(float))
    print(pd.concat({"lat": lat_delta, "lon": lon_delta}, axis=1).describe().round(4))

accident_weather.drop(columns=["event_day","wx_AccidentID","wx_weather_day"], errors='ignore', inplace=True)
accident_weather.to_pickle("data_sources/fused/accident_weather.pkl")
accident_weather.to_csv("data_sources/fused/accident_weather.csv", index=False)


# ## Matched Aircraft Data Fusion

def clean_text(s):
    """ Normalizzazione del testo: rimozione di caratteri speciali, lowercase e spazi extra. """
    return re.sub(r'\W+', ' ', str(s)).lower().strip()

# Pulizia dei dati
accident_weather['Vehicles.Model'] = accident_weather['Vehicles.Model'].apply(clean_text)
accident_weather['Vehicles.Make'] = accident_weather['Vehicles.Make'].apply(clean_text)

accident_weather_path = 'data_sources/fused/accident_weather.pkl'
matched_results_path = 'data_sources/binding/matched_results.csv'

accident_weather_df = pd.read_pickle(accident_weather_path)
matched_results_df = pd.read_csv(matched_results_path)

# Normalize casing for matching
matched_results_df['NtsbNumber'] = matched_results_df['NtsbNumber'].str.lower()
matched_results_df['EventDate'] = pd.to_datetime(matched_results_df['EventDate'], errors='coerce')
matched_results_df['Vehicles.SerialNumber'] = matched_results_df['Vehicles.SerialNumber'].str.lower()
matched_results_df['Vehicles.RegistrationNumber'] = matched_results_df['Vehicles.RegistrationNumber'].str.lower()
matched_results_df['Vehicles.Make'] = matched_results_df['Vehicles.Make'].str.lower()
matched_results_df['Vehicles.Model'] = matched_results_df['Vehicles.Model'].str.lower()

matched_results_df.drop(columns=["JW_Score","LEV_Score","Jac_Score","SimilarityScore","Matched_Aircraft_Model"], errors='ignore', inplace=True)



accident_weather_df['NtsbNumber'] = accident_weather_df['NtsbNumber'].str.lower()
accident_weather_df['EventDate'] = pd.to_datetime(accident_weather_df['EventDate'], errors='coerce')
accident_weather_df['Vehicles.SerialNumber'] = accident_weather_df['Vehicles.SerialNumber'].astype(str).str.lower()
accident_weather_df['Vehicles.RegistrationNumber'] = accident_weather_df['Vehicles.RegistrationNumber'].astype(str).str.lower()
accident_weather_df['Vehicles.Make'] = accident_weather_df['Vehicles.Make'].astype(str).str.lower()
accident_weather_df['Vehicles.Model'] = accident_weather_df['Vehicles.Model'].astype(str).str.lower()

accident_weather_df.drop(columns=["Vehicles.VehicleNumber"], errors='ignore', inplace=True)
accident_weather_df.rename(columns={"wx_time": "weather_time"}, inplace=True)

for key in accident_weather_df.columns:
    if key.startswith('wx_'):
        accident_weather_df.rename(columns={key: key[3:]}, inplace=True)

# Define the merge keys
merge_keys = ['NtsbNumber','EventDate','Vehicles.SerialNumber', 'Vehicles.RegistrationNumber', 'Vehicles.Make', 'Vehicles.Model']

# Perform the merge
fused_df = accident_weather_df.merge(
    df_matches,
    how='left',
    left_on=merge_keys,
    right_on=merge_keys
)

# Drop the duplicate matching columns from the right
for key in merge_keys:
    fused_df.drop(columns=[f"{key}_y"], errors='ignore', inplace=True)
    fused_df.rename(columns={f"{key}_x": key}, inplace=True)

# Save the resulting dataframe
fused_df.to_pickle('data_sources/fused/accident_weather_enriched.pkl')
fused_df.to_csv("data_sources/fused/accident_weather_enriched.csv", index=False)

# Compute matching stats
total_records = len(accident_weather)
matched_records = fused_df['engine_type'].notna().sum()
unmatched_records = total_records - matched_records
match_percentage = (matched_records / total_records) * 100

# Print statistics
print("Fusion complete. Enriched dataset saved to: data_sources/fused/accident_weather_enriched.pkl")
print("\n--- Matching Statistics ---")
print(f"Total records in original dataset: {total_records}")
print(f"Total records matched with binding CSV: {matched_records}")
print(f"Total unmatched records: {unmatched_records}")
print(f"Match percentage: {match_percentage:.2f}%")


# ## FINAL INTEGRATED SCHEMA
# fixing conflicts between `engine_count` and `Vehicles.NumberOfEngines`

# Convert columns to nullable integers
engine_count_int = fused_df['engine_count'].astype('Int64')
vehicle_engines = fused_df['Vehicles.NumberOfEngines'].astype('Int64')

# Rule 1: Fill NaNs in Vehicles.NumberOfEngines with engine_count
fused_df['Vehicles.NumberOfEngines'] = vehicle_engines.combine_first(engine_count_int)

# Rule 2: If Vehicles.NumberOfEngines == 0 and engine_count > 0 → trust engine_count
mask_replace_zero = (
    (fused_df['Vehicles.NumberOfEngines'] == 0) &
    (engine_count_int > 0)
)
fused_df.loc[mask_replace_zero, 'Vehicles.NumberOfEngines'] = engine_count_int[mask_replace_zero]

# Rule 3: Overwrite in case of real conflict (≠ 0 and ≠ each other)
conflict_mask = (
    engine_count_int.notna() &
    fused_df['Vehicles.NumberOfEngines'].notna() &
    (fused_df['Vehicles.NumberOfEngines'] != engine_count_int) &
    (fused_df['Vehicles.NumberOfEngines'] != 0) &
    (engine_count_int != 0)
)
fused_df.loc[conflict_mask, 'Vehicles.NumberOfEngines'] = engine_count_int[conflict_mask]

# Drop auxiliary column
fused_df.drop(columns=['engine_count'], inplace=True)

# Save cleaned and final dataset
final_pkl_path = 'data_sources/fused/aviation_accident_integrated.pkl'
final_csv_path = 'data_sources/fused/aviation_accident_integrated.csv'

fused_df.to_pickle(final_pkl_path)
fused_df.to_csv(final_csv_path, index=False)

print(f"Fusion complete. Cleaned dataset saved to:\n  • {final_pkl_path}\n  • {final_csv_path}")
print(f"{conflict_mask.sum()} engine count conflicts were resolved by trusting the 'engine_count' value.")

