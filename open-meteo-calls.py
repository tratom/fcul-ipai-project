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

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PATH = 'data_sources/'
PKL_FILE = PATH+"filtered/ntsb.pkl"
WEATHER_CACHE_FILE = PATH+"weather_results.json"
MAX_CALLS_PER_DAY = 10000
COST_PER_CALL = 1.9
REQUEST_TIMEOUT = 15                # Seconds to wait for server response
MAX_RETRIES_TIMEOUT = 3             # Retries for a single request on timeout
MAX_CONSECUTIVE_TIMEOUTS = 3        # If we fail 3 accidents in a row, we stop
SLEEP_ON_429_SECS = 60              # Wait time if we get a 429
RETRIES_ON_429 = 3                  # Number of times to retry after 429

# ------------------------------------------------------------------
# 1) Load the NTSB DataFrame
# ------------------------------------------------------------------
df_ntsb = pd.read_pickle(PKL_FILE)

unique_accidents = df_ntsb[["NtsbNumber","Latitude","Longitude","EventDate"]].drop_duplicates()

# ------------------------------------------------------------------
# 2) Load partial cached results (if any)
# ------------------------------------------------------------------
if os.path.exists(WEATHER_CACHE_FILE):
    with open(WEATHER_CACHE_FILE, 'r', encoding='utf-8') as f:
        weather_data_cache = json.load(f)
else:
    weather_data_cache = {}

calls_made_today = 0
consecutive_timeouts = 0  # Track consecutive accidents that fail all retries

# ------------------------------------------------------------------
# Helper function to perform the request with timeouts, retries, 429 handling
# ------------------------------------------------------------------
def fetch_weather_data(lat, lon, date_str, cache_key):
    """
    Attempt to fetch data up to MAX_RETRIES_TIMEOUT times if we get a timeout,
    and up to RETRIES_ON_429 times if we get status 429.
    Returns the 'hourly' data dict (or empty dict) if all attempts fail.
    Raises a custom "AllRetriesTimeout" exception if timeouts keep failing.
    """
    endpoint = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join([
            "temperature_2m","relative_humidity_2m","dew_point_2m","pressure_msl",
            "surface_pressure","precipitation","rain","snowfall","cloud_cover",
            "cloud_cover_low","cloud_cover_mid","cloud_cover_high","wind_speed_10m",
            "wind_speed_100m","wind_direction_10m","wind_direction_100m",
            "wind_gusts_10m","weather_code","snow_depth"
        ]),
        "timezone": "GMT"
    }

    timeout_attempts = 0

    while timeout_attempts < MAX_RETRIES_TIMEOUT:
        # We'll also keep track of 429 attempts
        status429_attempts = 0
        while True:
            try:
                # Perform the request with a timeout
                response = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
                # Check HTTP status
                if response.status_code == 200:
                    return response.json().get("hourly", {})
                elif response.status_code == 429:
                    status429_attempts += 1
                    if status429_attempts <= RETRIES_ON_429:
                        print(f"[{cache_key}] Got 429. Waiting {SLEEP_ON_429_SECS}s then retrying (attempt {status429_attempts}/{RETRIES_ON_429}).")
                        time.sleep(SLEEP_ON_429_SECS)
                        continue  # Retry the same request
                    else:
                        print(f"[{cache_key}] Too many 429 responses. Giving up this accident.")
                        return {}
                else:
                    # Some other error status code
                    print(f"[{cache_key}] Request failed: {response.status_code}. Skipping.")
                    return {}
            except requests.exceptions.Timeout:
                timeout_attempts += 1
                if timeout_attempts < MAX_RETRIES_TIMEOUT:
                    print(f"[{cache_key}] Timed out attempt {timeout_attempts}/{MAX_RETRIES_TIMEOUT}. Retrying.")
                else:
                    print(f"[{cache_key}] Timed out {MAX_RETRIES_TIMEOUT} times. Giving up.")
                    # We'll raise to handle the "consecutive timeouts" logic in the main loop
                    raise TimeoutError("All timeouts used up")
            except Exception as e:
                print(f"[{cache_key}] Unexpected error: {e}")
                return {}

    # If we exit the loop by normal means, we return empty
    return {}

# ------------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------------
try:
    for idx, row in unique_accidents.iterrows():
        ntsb_number = row["NtsbNumber"]
        date_str = pd.to_datetime(row["EventDate"]).strftime('%Y-%m-%d')
        lat = row["Latitude"]
        lon = row["Longitude"]
        cache_key = f"{ntsb_number}_{date_str}_{lat}_{lon}"

        # Skip if we already have data
        if cache_key in weather_data_cache:
            continue

        # Check daily limit
        if calls_made_today + COST_PER_CALL > MAX_CALLS_PER_DAY:
            print("Reached daily limit.")
            break

        # Attempt to fetch data
        try:
            data_hourly = fetch_weather_data(lat, lon, date_str, cache_key)
        except TimeoutError:
            # Means we timed out all attempts
            data_hourly = {}
            consecutive_timeouts += 1
            if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                print(f"Hit {MAX_CONSECUTIVE_TIMEOUTS} consecutive timeouts. Stopping the script.")
                break
        else:
            # If we didn't raise TimeoutError, reset consecutive timeouts
            consecutive_timeouts = 0

        # Store results (empty or not)
        weather_data_cache[cache_key] = data_hourly
        print(f"[{idx}] {cache_key} => {('SUCCESS' if data_hourly else 'FAILED')}")

        calls_made_today += COST_PER_CALL

        # Save partial results every 100
        if idx % 100 == 0:
            print(f"Processed {idx} accidents. Saving partial results.")
            with open(WEATHER_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(weather_data_cache, f)

except KeyboardInterrupt:
    print("\nInterrupted by user (Ctrl+C). Saving data and exiting...")

finally:
    # Always save final results to disk
    with open(WEATHER_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(weather_data_cache, f)
    print("Data saved to", WEATHER_CACHE_FILE)
    print("Done.")