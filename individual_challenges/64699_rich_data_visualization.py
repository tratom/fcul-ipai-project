"""
NTSB x Weather x Airline-Traffic Dashboard
=========================================
Author: Tommaso Tragno, 64699 - Rich Data Visualizations & Insights challenge
Updated: 2025-05-25

A Dash application to explore U.S. NTSB aviation accidents, local weather
observations and monthly airline-traffic statistics (passengers,
flights and load-factor).

Getting started
---------------
pip install pandas dash dash-bootstrap-components plotly
python 64699_rich_data_visualization.py

Then open http://127.0.0.1:8050/
"""
from __future__ import annotations

import functools
import pathlib
from datetime import date, datetime
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

###############################################################################
# DataFrame Paths                                                              #
###############################################################################

DATA_DIR = pathlib.Path("data_sources/filtered")
NTSB_FILE = DATA_DIR / "ntsb.pkl"
WEATHER_FILE = DATA_DIR / "weather.pkl"
TRAFFIC_FILE = DATA_DIR / "airline.pkl"

###############################################################################
# Data loading                                                                 #
###############################################################################

def _read_pickle(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_pickle(path)

@functools.lru_cache(None)
def load_data():
    return _read_pickle(NTSB_FILE), _read_pickle(WEATHER_FILE), _read_pickle(TRAFFIC_FILE)

###############################################################################
# Pre-processing                                                               #
###############################################################################

def preprocess(ntsb_df: pd.DataFrame, wx_df: pd.DataFrame, traf_df: pd.DataFrame):
    """Produce ready-to-plot tables for accidents, weather and traffic."""
    # ── Accident data ─────────────────────────────────────────────────────
    ntsb = ntsb_df.copy()
    ntsb["EventDate"] = pd.to_datetime(ntsb["EventDate"])
    ntsb["date"] = ntsb["EventDate"].dt.date

    # ── Weather at accident hour ──────────────────────────────────────────
    wx = wx_df.copy()
    wx["time"] = pd.to_datetime(wx["time"])
    wx["NtsbNumber"] = wx["AccidentID"].str.split("_").str[0]
    wx["date"] = wx["time"].dt.date

    wx_event = (
        wx.merge(ntsb[["NtsbNumber", "EventDate"]], on="NtsbNumber", how="inner")
          .loc[lambda d: d["time"].dt.floor("h") == d["EventDate"].dt.floor("h")]
          .drop_duplicates("NtsbNumber")
    )
    wx_event["precip_total"] = wx_event["rain"].fillna(0) + wx_event["snowfall"].fillna(0)

    ntsb = ntsb.merge(wx_event[["NtsbNumber", "temperature_2m", "wind_speed_10m", "precip_total"]], on="NtsbNumber", how="left")

    # ── Daily join for the bar/line view ──────────────────────────────────
    daily_acc = ntsb.groupby("date").size().to_frame(name="accidents").reset_index()
    daily_temp = wx.groupby("date")["temperature_2m"].mean().reset_index(name="avg_temp")
    daily = daily_acc.merge(daily_temp, on="date", how="left")

    # ── Airline traffic - normalise month key ─────────────────────────────
    traf = traf_df.copy()
    traf["month"] = pd.to_datetime(traf[["Year", "Month"]].assign(Day=1))
    traf["month"] = traf["month"].dt.to_period("M").dt.to_timestamp()
    traf.rename(columns={"Pax": "passengers", "Flt": "flights", "LF": "load_factor"}, inplace=True)

    ntsb["month"] = ntsb["EventDate"].dt.to_period("M").dt.to_timestamp()
    acc_month = ntsb.groupby("month").size().reset_index(name="accidents")

    traffic = traf.merge(acc_month, on="month", how="left")
    traffic["accidents"] = traffic["accidents"].fillna(0)
    traffic["acc_per_million"] = traffic["accidents"] / (traffic["passengers"] / 1_000_000)

    return {"ntsb": ntsb, "daily": daily, "traffic": traffic}

###############################################################################
# Plot helpers                                                                 #
###############################################################################

def empty_fig(msg="No data"):
    return go.Figure().update_layout(title_text=msg, height=500, template="plotly_white")

# ── Map ─────────────────────────────────────────────────────────────────────

def build_map(df: pd.DataFrame, max_points_sample=3000) -> go.Figure:
    """Scatter map (MapLibre) with clustering, stable colours & reliable refresh."""
    if df.empty:
        return empty_fig()

    dm = df.dropna(subset=["Latitude", "Longitude"]).copy()
    dm["_size"] = dm["TotalInjuryCount"].fillna(0).astype(int).clip(lower=1)

    dm["HighestInjury"] = dm["HighestInjury"].str.title()

    cat_order = ["Fatal", "Serious", "Minor", "None", "Unknown"]
    colour_map = {
        "Fatal": "#d62728",
        "Serious": "#ff7f0e",
        "Minor": "#1f77b4",
        "None": "#2ca02c",
        "Unknown": "#9467bd",
    }
    dm["HighestInjury"] = pd.Categorical(dm["HighestInjury"],
                                     categories=cat_order,
                                     ordered=True)

    # light down‑sample before sending to the browser (clustering still reports true counts)
    if len(dm) > max_points_sample:
        dm = dm.sample(max_points_sample, random_state=0)

    fig = px.scatter_mapbox(
        dm,
        lat="Latitude",
        lon="Longitude",
        size="_size",
        color="HighestInjury",
        size_max=20,
        zoom=3,
        height=450,
        color_discrete_map=colour_map,
        category_orders={"HighestInjury": cat_order},
        hover_name="NtsbNumber",
    )

    # Enable client‑side clustering for snappy zooming
    fig.update_traces(cluster=dict(enabled=True, maxzoom=8, step=50))

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        # unique value so Dash always forces a full redraw when data changes
        uirevision=str(pd.Timestamp.utcnow().value),
    )
    return fig

# ── Weather scatters ─────────────────────────────────────────────────────────

def build_scatter_wind(df):
    if df.empty: return empty_fig()
    return px.scatter(df, x="wind_speed_10m", y="temperature_2m", color=df["TotalInjuryCount"].fillna("Unknown"), opacity=0.6,
                      trendline="ols", labels={"wind_speed_10m":"Wind (m/s)","temperature_2m":"Temp (°C)"}, height=500,
                      title="Wind vs Temp")

def build_scatter_precip(df):
    if df.empty: return empty_fig()
    return px.scatter(df, x="precip_total", y="temperature_2m", color=df["TotalInjuryCount"].fillna("Unknown"), opacity=0.6,
                      trendline="ols", labels={"precip_total":"Precip (mm)","temperature_2m":"Temp (°C)"}, height=500,
                      title="Precipitation vs Temp")

# ── Daily accidents + avg temp bar/line  ─────────────────────────────────────

def build_timeseries(daily, start, end):
    sel = daily[(daily["date"]>=start)&(daily["date"]<=end)].copy()
    if sel.empty: return empty_fig()
    sel.sort_values("date", inplace=True)
    fig = px.bar(sel, x="date", y="accidents", labels={"accidents":"Accidents"}, height=500)
    fig.add_trace(px.line(sel, x="date", y="avg_temp").data[0])
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return fig

# ── Monthly traffic line (dual axis) ─────────────────────────────────────────

def build_traffic_line(df):
    if df.empty:
        return empty_fig()
    fig = go.Figure()
    fig.add_bar(x=df["month"], y=df["accidents"], name="Accidents", yaxis="y")
    fig.add_scatter(x=df["month"], y=df["passengers"], name="Passengers", yaxis="y2")
    fig.update_layout(title="Monthly accidents vs passengers", height=500,
                      yaxis=dict(title="Accidents"),
                      yaxis2=dict(title="Passengers", overlaying="y", side="right"))
    return fig


def build_traffic_scatter(df):
    if df.empty:
        return empty_fig()
    return px.scatter(df, x="load_factor", y="acc_per_million", size="flights", color="flights", height=500,
                      labels={"load_factor":"Load factor (%)","acc_per_million":"Accidents per M pax"},
                      title="Safety vs Load factor")

# ── Other plots (heat, bar, box) ─────────────────────────────────────────────

def build_heat(df):
    if df.empty: return empty_fig()
    bins = pd.cut(df["temperature_2m"], [-50,-20,-10,0,5,10,15,20,25,30,40], right=False)
    hm = df.assign(hour=df["EventDate"].dt.hour, tbin=bins).groupby(["hour","tbin"]).size().reset_index(name="n")
    hm["tbin"] = hm["tbin"].astype(str)
    return px.density_heatmap(hm, x="hour", y="tbin", z="n", color_continuous_scale="Inferno", height=500,
                              labels={"hour":"Hour","tbin":"Temp","n":"Accidents"})

def build_bar(df):
    if df.empty: return empty_fig()
    top = df["Vehicles.Model"].dropna().value_counts().head(10).rename_axis("Model").reset_index(name="Count")
    return px.bar(top, x="Model", y="Count", height=500, title="Top 10 aircraft models").update_layout(xaxis_tickangle=-45)

def build_box(df):
    if df.empty: return empty_fig()
    return px.box(df, x="HighestInjury", y="temperature_2m", points="all", height=500, title="Temp by injury")

###############################################################################
# Filtering helpers                                                            #
###############################################################################

def filter_acc(df, start, end, makes, injuries):
    m = (df["date"]>=start)&(df["date"]<=end)
    if makes: m &= df["Vehicles.Make"].isin(makes)
    if injuries: m &= df["HighestInjury"].isin(injuries)
    return df[m]

###############################################################################
# Layout                                                                       #
###############################################################################

def controls(df):
    min_d, max_d = df["date"].min(), df["date"].max()
    make_opts = [{"label":m.title(),"value":m} for m in sorted(df["Vehicles.Make"].dropna().unique())]
    injury_opts = [{"label":c,"value":c} for c in df["HighestInjury"].cat.categories]
    return dbc.Card([
        dbc.CardHeader(html.H5("Filters")),
        dbc.CardBody([
            html.Label("Date range"),
            dcc.DatePickerRange(id="dates", start_date=min_d, end_date=max_d, min_date_allowed=min_d, max_date_allowed=max_d),
            html.Hr(),
            html.Label("Aircraft make"), dcc.Dropdown(id="make", options=make_opts, multi=True),
            html.Label("Highest injury", className="mt-3"), dcc.Checklist(id="inj", options=injury_opts, value=list(c for c in df["HighestInjury"].cat.categories)),
        ])
    ], className="h-100")


def layout(prep):
    return dbc.Container([
        html.H2("Aviation Accidents Data Visualization", className="my-3"),
        dbc.Row([
            dbc.Col(controls(prep["ntsb"]), width=3),
            dbc.Col(dcc.Graph(id="map"), width=9),
        ]),
        dbc.Tabs([
            dbc.Tab(dcc.Graph(id="wind"), label="Wind vs Temp"),
            dbc.Tab(dcc.Graph(id="precip"), label="Precip vs Temp"),
            dbc.Tab(dcc.Graph(id="ts"), label="Daily Accidents & Temp"),
            dbc.Tab(dcc.Graph(id="heat"), label="Hourly Heat"),
            dbc.Tab(dcc.Graph(id="bar"), label="Models"),
            dbc.Tab(dcc.Graph(id="box"), label="Weather by Injury"),
            dbc.Tab(dcc.Graph(id="traf-line"), label="Traffic trends"),
            dbc.Tab(dcc.Graph(id="traf-scatter"), label="Traffic scatter"),
        ], className="mt-3"),
    ], fluid=True)

###############################################################################
# Dash app                                                                     #
###############################################################################

def create_app():
    warnings.filterwarnings("ignore", category=FutureWarning)
    ntsb, wx, traf = load_data()
    prep = preprocess(ntsb, wx, traf)

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = layout(prep)

    @app.callback(
        Output("map", "figure"), Output("wind", "figure"), Output("precip", "figure"),
        Output("ts", "figure"), Output("heat", "figure"), Output("bar", "figure"), Output("box", "figure"),
        Output("traf-line", "figure"), Output("traf-scatter", "figure"),
        Input("dates", "start_date"), Input("dates", "end_date"), Input("make", "value"), Input("inj", "value"),
    )
    def update(start, end, make, inj):
        s = datetime.fromisoformat(start).date() if start else prep["ntsb"]["date"].min()
        e = datetime.fromisoformat(end).date() if end else prep["ntsb"]["date"].max()
        df_sel = filter_acc(prep["ntsb"], s, e, make, inj)
        # traffic range filter - convert start/end to month first day
        traf_sel = prep["traffic"].loc[(prep["traffic"]["month"]>=pd.to_datetime(s).replace(day=1)) & (prep["traffic"]["month"]<=pd.to_datetime(e).replace(day=1))]
        return (
            build_map(df_sel),
            build_scatter_wind(df_sel),
            build_scatter_precip(df_sel),
            build_timeseries(prep["daily"], s, e),
            build_heat(df_sel),
            build_bar(df_sel),
            build_box(df_sel),
            build_traffic_line(traf_sel),
            build_traffic_scatter(traf_sel),
        )

    return app

if __name__ == "__main__":
    create_app().run(debug=False)
