# app.py
import os
import io
from datetime import timedelta

import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

# Forecasting libs
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Optional deep learning & Prophet 
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except Exception:
    Sequential = None
    LSTM = None
    Dense = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="HeatCast: Heatwave Risk Analysis", layout="wide")
STATES = ["Sokoto", "Kebbi", "Katsina", "Jigawa", "Zamfara"]
DATA_DIR = "data"

# -----------------------------
# Column labels (human-readable)
# -----------------------------
COLUMN_LABELS = {
    "name": "Station Name",
    "datetime": "Date/Time",
    "tempmax": "Maximum Temperature (°C)",
    "tempmin": "Minimum Temperature (°C)",
    "temp": "Average Temperature (°C)",
    "humidity": "Relative Humidity (%)",
    "precip": "Precipitation (mm)",
    "windspeed": "Wind Speed (km/h)",
    "windgust": "Wind Gust (km/h)",
    "hi": "Heat Index (°C)"  # derived
}

USE_COLS = [
    "name", "datetime", "tempmax", "tempmin", "temp",
    "humidity", "precip", "windspeed", "windgust"
]

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_state_data(state: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{state} Climate Data - 2010-Nov-2025.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found for {state}: {path}")
    df = pd.read_csv(path, low_memory=False)
    # Select only relevant columns if present
    cols_present = [c for c in USE_COLS if c in df.columns]
    df = df[cols_present].copy()
    df = standardize_columns(df)
    df = clean_units(df)
    df["state"] = state
    df = compute_heat_index_c(df)
    df = add_features(df)
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize column names to lower snake case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Ensure datetime exists
    if "datetime" not in df.columns:
        raise ValueError("No datetime column found (expected 'datetime').")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    # Ensure temperature columns
    if "temp" not in df.columns and {"tempmax","tempmin"}.issubset(df.columns):
        df["temp"] = (df["tempmax"] + df["tempmin"]) / 2.0
    return df

#def clean_units(df: pd.DataFrame) -> pd.DataFrame:
 #   df = df.copy()
  ## for col in ["tempmax", "tempmin", "temp"]:
    #    if col in df.columns:
     #       df[col] = df[col].clip(lower=-10, upper=60)
    #if "humidity" in df.columns:
        #df["humidity"] = df["humidity"].clip(lower=0, upper=100)
    #if "precip" in df.columns:
        #df["precip"] = df["precip"].clip(lower=0)
    # Windspeed/gust are in km/h per schema; keep as is for display, convert internally if needed
    #return df

def clean_units(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert Fahrenheit → Celsius first
    for col in ["tempmax", "tempmin", "temp"]:
        if col in df.columns:
            df[col] = (df[col] - 32) * 5.0/9.0

    # Mask unrealistic anomalies instead of clipping
    for col in ["tempmax", "tempmin", "temp"]:
        if col in df.columns:
            df[col] = df[col].mask((df[col] < -10) | (df[col] > 55), np.nan)

    # Humidity sanity check
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(lower=0, upper=100)

    # Precipitation sanity check
    if "precip" in df.columns:
        df["precip"] = df["precip"].clip(lower=0)

    # Windspeed/gust are in km/h per schema; keep as is
    return df


# Convert units
#def convert_to_celsius(df):
    #for col in ["tempmax", "tempmin", "temp"]:
        #if col in df.columns:
            #df[col] = (df[col] - 32) * 5.0/9.0
    #return df


def compute_heat_index_c(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Use tempmax and humidity for daily HI approximation
    if not {"tempmax", "humidity"}.issubset(df.columns):
        df["hi"] = np.nan
        return df
    T_c = df["tempmax"].astype(float)
    RH = df["humidity"].astype(float)
    T_f = T_c * 9/5 + 32
    
    # Rothfusz regression (valid for T_f >= 80F)
    HI_f = (
        -42.379 + 2.04901523*T_f + 10.14333127*RH
        - 0.22475541*T_f*RH - 0.00683783*T_f**2
        - 0.05481717*RH**2 + 0.00122874*T_f**2*RH
        + 0.00085282*T_f*RH**2 - 0.00000199*T_f**2*RH**2
    )
    HI_f = np.where(T_f < 80, T_f, HI_f)
    df["hi"] = (HI_f - 32) * 5/9
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["doy"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    # Lags
    for col in ["tempmax", "hi", "temp"]:
        if col in df.columns:
            for lag in [1,2,3,7,14]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    # Rolling
    if "tempmax" in df.columns:
        df["tempmax_roll7"] = df["tempmax"].rolling(7).mean()
        df["tempmax_roll14"] = df["tempmax"].rolling(14).mean()
    if "hi" in df.columns:
        df["hi_roll7"] = df["hi"].rolling(7).mean()
    return df

def compute_threshold(df: pd.DataFrame, col="tempmax", pct=0.95) -> float:
    return df[col].quantile(pct)

def detect_heatwaves(df: pd.DataFrame, col="tempmax", thr=None, min_consec=3):
    df = df.copy()
    if thr is None:
        thr = compute_threshold(df, col=col, pct=0.95)
    df["thr"] = thr
    df["hot_day"] = (df[col] >= df["thr"]).astype(int)
    df["exceed"] = (df[col] - df["thr"]).clip(lower=0)

    # Label consecutive events
    df = df.sort_values("datetime")
    runs = []
    run_start = None
    run_len = 0
    for i, row in df.iterrows():
        if row["hot_day"] == 1:
            if run_len == 0:
                run_start = row["datetime"]
            run_len += 1
        else:
            if run_len >= min_consec:
                runs.append((run_start, df.loc[i-1, "datetime"], run_len))
            run_len = 0
    if run_len >= min_consec:
        runs.append((run_start, df.iloc[-1]["datetime"], run_len))

    df["event_id"] = np.nan
    eid = 0
    for start, end, length in runs:
        mask = (df["datetime"] >= start) & (df["datetime"] <= end)
        df.loc[mask, "event_id"] = eid
        eid += 1

    events = (
        df.dropna(subset=["event_id"])
          .groupby("event_id")
          .agg(start=("datetime","min"), end=("datetime","max"),
               duration=("datetime","count"),
               peak_intensity=("exceed","max"),
               mean_intensity=("exceed","mean"))
          .reset_index()
    )
    return df, events

def narrative_blocks(df: pd.DataFrame, col="tempmax", thr=None):
    latest = df.iloc[-30:].copy() if len(df) >= 30 else df.copy()
    mean_val = latest[col].mean()
    max_val = latest[col].max()
    if thr is None:
        thr = compute_threshold(df, col)
    hot_share = (latest[col] >= thr).mean() if len(latest) else 0

    observation = (
        f"Recent {COLUMN_LABELS.get(col, col)} averages are around {mean_val:.1f}°C, "
        f"with peaks up to {max_val:.1f}°C. "
        f"Approximately {hot_share*100:.0f}% of the last 30 days exceeded the local threshold."
    )
    implication = (
        "Sustained high temperatures elevate heat stress, increasing risks of dehydration, "
        "heat exhaustion, and productivity losses—especially for outdoor workers and vulnerable groups."
    )
    recommendation = (
        "Issue targeted advisories during high‑risk windows, expand access to cooling and water points, "
        "and adjust work schedules to early morning/evening. Coordinate health surveillance for heat‑related illness."
    )
    return observation, implication, recommendation

# -----------------------------
# Forecasting helpers
# -----------------------------
def prepare_supervised(df: pd.DataFrame, target="tempmax"):
    feat_cols = [c for c in df.columns if ("lag" in c and any(k in c for k in [target, "hi", "temp"]))] + ["doy","month"]
    df_sup = df.dropna(subset=feat_cols+[target]).copy()
    X = df_sup[feat_cols]
    y = df_sup[target]
    return df_sup, X, y, feat_cols

def forecast_ml(df: pd.DataFrame, model_name="XGBoost", target="tempmax", horizon=7):
    df_sup, X, y, feat_cols = prepare_supervised(df, target)
    if model_name == "XGBoost":
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
    elif model_name == "LightGBM":
        model = LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=-1, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42)
    else:
        model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    # Iterative forecast using last known lags
    last_date = df["datetime"].max()
    tail = df_sup.copy()
    rows = []
    for h in range(1, horizon+1):
        row = {"datetime": last_date + timedelta(days=h)}
        row["doy"] = row["datetime"].timetuple().tm_yday
        row["month"] = row["datetime"].month
        for col in ["tempmax","hi","temp"]:
            for lag in [1,2,3,7,14]:
                lag_col = f"{col}_lag{lag}"
                if lag_col in tail.columns:
                    row[lag_col] = tail.iloc[-lag][col] if len(tail) >= lag else np.nan
        feat = pd.DataFrame([row])[feat_cols]
        pred = model.predict(feat)[0]
        row[target] = pred
        rows.append(row)
        # Append predicted row to update lags
        new_row = {**row}
        for lag in [1,2,3,7,14]:
            new_row[f"{target}_lag{lag}"] = tail.iloc[-lag][target] if len(tail) >= lag else np.nan
        tail = pd.concat([tail, pd.DataFrame([new_row])], ignore_index=True)
    fc = pd.DataFrame(rows)
    return fc, model

def forecast_arima(df: pd.DataFrame, target="tempmax", horizon=7, order=(2,1,2)):
    series = df.set_index("datetime")[target].dropna()
    model = ARIMA(series, order=order)
    res = model.fit()
    fc = res.get_forecast(steps=horizon)
    fdf = pd.DataFrame({
        "datetime": pd.date_range(series.index.max() + timedelta(days=1), periods=horizon, freq="D"),
        target: fc.predicted_mean.values
    })
    return fdf, res

def forecast_sarima(df: pd.DataFrame, target="tempmax", horizon=7, order=(2,1,2), seasonal_order=(1,1,1,7)):
    series = df.set_index("datetime")[target].dropna()
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    fdf = pd.DataFrame({
        "datetime": pd.date_range(series.index.max() + timedelta(days=1), periods=horizon, freq="D"),
        target: fc.predicted_mean.values
    })
    return fdf, res

def forecast_prophet(df: pd.DataFrame, target="tempmax", horizon=7):
    if Prophet is None:
        raise RuntimeError("Prophet not available in environment.")
    d = df[["datetime", target]].dropna().rename(columns={"datetime":"ds", target:"y"})
    m = Prophet(seasonality_mode="additive", yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(d)
    future = m.make_future_dataframe(periods=horizon, freq="D")
    fc = m.predict(future).tail(horizon)[["ds","yhat"]].rename(columns={"ds":"datetime","yhat":target})
    return fc, m

def forecast_lstm(df: pd.DataFrame, target="tempmax", horizon=7, lookback=30):
    if Sequential is None or LSTM is None:
        raise RuntimeError("TensorFlow/Keras not available in environment.")
    series = df[target].dropna().values.astype("float32")
    # Build supervised sequences
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    model = Sequential([LSTM(32, input_shape=(lookback,1)), Dense(1)])
    model.compile(optimizer="adam", loss="mae")
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    # Forecast iteratively
    last_seq = series[-lookback:].reshape(1, lookback, 1)
    preds = []
    for _ in range(horizon):
        p = model.predict(last_seq, verbose=0)[0,0]
        preds.append(p)
        last_seq = np.roll(last_seq, -1, axis=1)
        last_seq[0, -1, 0] = p
    fdf = pd.DataFrame({
        "datetime": pd.date_range(df["datetime"].max() + timedelta(days=1), periods=horizon, freq="D"),
        target: preds
    })
    return fdf, model

def compute_risk(forecast_df: pd.DataFrame, hist_df: pd.DataFrame, target="tempmax", pct=0.95, w=(0.5,0.2,0.3)):
    thr = hist_df[target].quantile(pct)
    df = forecast_df.copy()
    df["thr"] = thr
    df["exceed"] = (df[target] - df["thr"]).clip(lower=0)
    df = df.sort_values("datetime")
    df["hot_forecast"] = (df[target] >= df["thr"]).astype(int)
    df["duration_proxy"] = df["hot_forecast"].rolling(3).sum().fillna(0)
    df["prob_proxy"] = (df["exceed"] / (abs(df["thr"]) + 1e-6)).clip(0,1)
    a,b,c = w
    df["risk_score_raw"] = (a*df["exceed"] + b*df["duration_proxy"] + c*df["prob_proxy"])
    # Normalize 0–100
    df["risk_score"] = 100 * (df["risk_score_raw"] - df["risk_score_raw"].min()) / (df["risk_score_raw"].max() - df["risk_score_raw"].min() + 1e-6)
    return df

def html_summary(state, events, forecast_df, risk_df):
    buf = io.StringIO()
    buf.write(f"<h2>HeatCast Summary — {state}</h2>")
    buf.write("<h3>Detected Heatwave Events</h3>")
    buf.write(events.to_html(index=False))
    buf.write("<h3>Forecast</h3>")
    buf.write(forecast_df.to_html(index=False))
    buf.write("<h3>Risk Scores</h3>")
    buf.write(risk_df.to_html(index=False))
    return buf.getvalue()

# -----------------------------
# Sidebar
# -----------------------------
#with st.sidebar:
    #st.title("HeatCast Controls")
    #state = st.selectbox("Select state", STATES, index=0)
    #pct = st.slider("Heatwave percentile threshold (Maximum Temperature)", 0.80, 0.99, 0.95, 0.01)
    #min_consec = st.slider("Minimum consecutive hot days", 2, 7, 3, 1)
    #horizon = st.select_slider("Forecast horizon (days)", options=[7,14,30], value=7)
    #target_var = st.selectbox("Target variable", ["tempmax", "hi", "temp"], index=0, format_func=lambda x: COLUMN_LABELS.get(x, x))

# -----------------------------
# Navigation
# -----------------------------
selected = option_menu(
    menu_title="HeatCast",
    options=["Overview", "Analysis", "Forecast", "Heat Risk", "About"],
    icons=["house", "bar-chart", "graph-up", "exclamation-triangle", "info-circle"],
    orientation="horizontal"
)

# -----------------------------
# Overview
# -----------------------------
if selected == "Overview":
    st.header("HeatCast: Heatwave Risk Analysis and Forecasting")
    st.markdown(f"""
**Scope:** Localized heatwave analytics and short‑term forecasting for five major states in Northeast Nigeria.

**What you can do:**
- **Explore climate trends**: Univariate, bivariate, and multivariate analysis of temperature, humidity, wind, and precipitation.
- **Detect heatwaves**: Threshold‑based and Heat Index–based event catalog with duration and intensity.
- **Forecast heat stress**: Choose SARIMA, ARIMA, XGBoost, LightGBM, LSTM, or Prophet for 7/14/30‑day forecasts.
- **Assess risk**: State‑level risk scores, forecast bands, event timelines, and actionable recommendations.
- **Export**: CSVs for events and forecasts, plus an HTML summary suitable for PDF conversion.
""")

# -----------------------------
# Analysis
# -----------------------------
elif selected == "Analysis":
    
    # -----------------------------
    # Sidebar
    # -----------------------------
    with st.sidebar:
        st.title("HeatCast Controls")
        state = st.selectbox("Select state", STATES, index=0)
        pct = st.slider("Heatwave percentile threshold (Maximum Temperature)", 0.80, 0.99, 0.95, 0.01)
        min_consec = st.slider("Minimum consecutive hot days", 2, 7, 3, 1)
        horizon = st.select_slider("Forecast horizon (days)", options=[7,14,30], value=7)
        target_var = st.selectbox("Target variable", ["tempmax", "hi", "temp"], index=0, format_func=lambda x: COLUMN_LABELS.get(x, x))

    # Load data once per state
    df = load_state_data(state)
    
    st.header(f"Analysis — {state}")

    # Univariate
    st.subheader("Univariate analysis")
    cols_uni = ["tempmax","tempmin","temp","hi","humidity","precip","windspeed","windgust"]
    cols_uni = [c for c in cols_uni if c in df.columns]
    metric = st.selectbox("Select metric", cols_uni, index=0, format_func=lambda x: COLUMN_LABELS.get(x, x))

    fig = px.line(
        df, x="datetime", y=metric,
        title=f"{state}: {COLUMN_LABELS.get(metric, metric)} over time",
        labels={"datetime": COLUMN_LABELS["datetime"], metric: COLUMN_LABELS.get(metric, metric)}
    )
    st.plotly_chart(fig, use_container_width=True)

    thr = compute_threshold(df, col="tempmax", pct=pct)
    obs, imp, rec = narrative_blocks(df, col=metric, thr=thr)
    st.markdown(f"**Observation:** {obs}")
    st.markdown(f"**Implications:** {imp}")
    st.markdown(f"**Recommendations:** {rec}")

    # Bivariate
    st.subheader("Bivariate analysis")
    x_var = st.selectbox("X variable", cols_uni, index=0, key="xvar", format_func=lambda x: COLUMN_LABELS.get(x, x))
    y_var = st.selectbox("Y variable", cols_uni, index=min(1, len(cols_uni)-1), key="yvar", format_func=lambda x: COLUMN_LABELS.get(x, x))
    fig2 = px.scatter(
        df, x=x_var, y=y_var, color=df["month"].astype(str), trendline="ols",
        title=f"{state}: {COLUMN_LABELS.get(x_var, x_var)} vs {COLUMN_LABELS.get(y_var, y_var)}",
        labels={x_var: COLUMN_LABELS.get(x_var, x_var), y_var: COLUMN_LABELS.get(y_var, y_var)}
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Observation:** Seasonal clustering indicates strong cycles influencing the relationship between variables.")
    st.markdown("**Implications:** Compound risk arises when high temperature coincides with high humidity and low wind.")
    st.markdown("**Recommendations:** Align advisories with seasonal windows; prioritize cooling access and hydration during peak coupling periods.")

    # Multivariate
    st.subheader("Multivariate analysis")
    mv_cols = [c for c in ["tempmax","hi","humidity","windspeed","precip","temp"] if c in df.columns]
    corr = df[mv_cols].corr()
    fig3 = px.imshow(
        corr, text_auto=True, aspect="auto",
        title=f"{state}: Correlation matrix",
        labels={"color": "Correlation"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Observation:** Humidity and wind modulate perceived heat (Heat Index) beyond raw temperature.")
    st.markdown("**Implications:** Policy should consider compound indicators to capture real heat stress.")
    st.markdown("**Recommendations:** Integrate Heat Index into early warnings; coordinate shade, ventilation, and water provisioning.")

    # Extremes by month/year
    st.subheader("Extreme event frequency")
    threshold_extreme = st.slider("Extreme temperature threshold (°C)", 35.0, 45.0, 40.0, 0.5)
    df["is_extreme"] = (df["tempmax"] >= threshold_extreme).astype(int)
    extreme_month = df.groupby("month")["is_extreme"].sum().reset_index()
    fig_ext = px.bar(extreme_month, x="month", y="is_extreme", title=f"{state}: Extreme days per month (≥{threshold_extreme}°C)")
    st.plotly_chart(fig_ext, use_container_width=True)
    st.markdown("**Observation:** Concentration of extreme days in specific months highlights seasonal heatwave windows.")
    st.markdown("**Implications:** Resource allocation (cooling centers, water points) should be timed to these windows.")
    st.markdown("**Recommendations:** Pre‑position supplies and advisories ahead of peak months.")

# -----------------------------
# Forecast
# -----------------------------
elif selected == "Forecast":
    # -----------------------------
    # Sidebar
    # -----------------------------
    with st.sidebar:
        st.title("HeatCast Controls")
        state = st.selectbox("Select state", STATES, index=0)
        pct = st.slider("Heatwave percentile threshold (Maximum Temperature)", 0.80, 0.99, 0.95, 0.01)
        min_consec = st.slider("Minimum consecutive hot days", 2, 7, 3, 1)
        horizon = st.select_slider("Forecast horizon (days)", options=[7,14,30], value=7)
        target_var = st.selectbox("Target variable", ["tempmax", "hi", "temp"], index=0, format_func=lambda x: COLUMN_LABELS.get(x, x))

    # Load data once per state
    df = load_state_data(state)

    st.header(f"Forecast — {state}")
    model_choice = st.selectbox("Model", ["SARIMA","ARIMA","XGBoost","LightGBM","LSTM","Prophet"], index=2)

    try:
        if model_choice == "SARIMA":
            fc, model = forecast_sarima(df, target=target_var, horizon=horizon)
        elif model_choice == "ARIMA":
            fc, model = forecast_arima(df, target=target_var, horizon=horizon)
        elif model_choice == "XGBoost":
            fc, model = forecast_ml(df, model_name="XGBoost", target=target_var, horizon=horizon)
        elif model_choice == "LightGBM":
            fc, model = forecast_ml(df, model_name="LightGBM", target=target_var, horizon=horizon)
        elif model_choice == "LSTM":
            fc, model = forecast_lstm(df, target=target_var, horizon=horizon)
        elif model_choice == "Prophet":
            fc, model = forecast_prophet(df, target=target_var, horizon=horizon)
        else:
            fc, model = forecast_ml(df, model_name="GBR", target=target_var, horizon=horizon)

        # Plot
        hist_tail = df[["datetime", target_var]].tail(120)
        fc_all = pd.concat([hist_tail, fc], ignore_index=True)
        fig = px.line(
            fc_all, x="datetime", y=target_var,
            color=(fc_all["datetime"] > df["datetime"].max()).map({True:"Forecast", False:"History"}),
            title=f"{state}: {COLUMN_LABELS.get(target_var, target_var)} forecast ({model_choice}, {horizon} days)",
            labels={"datetime": COLUMN_LABELS["datetime"], target_var: COLUMN_LABELS.get(target_var, target_var)}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Narrative
        thr = compute_threshold(df, col="tempmax", pct=pct)
        obs, imp, rec = narrative_blocks(pd.concat([df, fc_all], ignore_index=True), col=target_var, thr=thr)
        st.markdown(f"**Observation:** {obs}")
        st.markdown(f"**Implications:** {imp}")
        st.markdown(f"**Recommendations:** {rec}")

        # Export forecast
        st.download_button(
            "Download forecast CSV",
            fc.to_csv(index=False).encode("utf-8"),
            file_name=f"{state.lower()}_{target_var}_forecast_{horizon}d.csv"
        )

    except Exception as e:
        st.error(f"Forecast error: {e}")

# -----------------------------
# Heat Risk
# -----------------------------
elif selected == "Heat Risk":

    # -----------------------------
    # Sidebar
    # -----------------------------
    with st.sidebar:
        st.title("HeatCast Controls")
        state = st.selectbox("Select state", STATES, index=0)
        pct = st.slider("Heatwave percentile threshold (Maximum Temperature)", 0.80, 0.99, 0.95, 0.01)
        min_consec = st.slider("Minimum consecutive hot days", 2, 7, 3, 1)
        horizon = st.select_slider("Forecast horizon (days)", options=[7,14,30], value=7)
        target_var = st.selectbox("Target variable", ["tempmax", "hi", "temp"], index=0, format_func=lambda x: COLUMN_LABELS.get(x, x))

        st.header(f"Heat Risk — {state}")
    
    # Load data once per state
    df = load_state_data(state)
    
    # Detect historical heatwaves
    df_hw, events = detect_heatwaves(df, col="tempmax", thr=compute_threshold(df, col="tempmax", pct=pct), min_consec=min_consec)
    st.subheader("Detected heatwave events")
    st.dataframe(events)

    # Forecast bands for 7/14/30
    horizons = [7,14,30]
    bands = []
    for h in horizons:
        fc_h, _ = forecast_ml(df, model_name="LightGBM", target=target_var, horizon=h)
        risk_h = compute_risk(fc_h, df, target=target_var, pct=pct)
        fc_h["horizon"] = h
        risk_h["horizon"] = h
        bands.append((fc_h, risk_h))

    # Plot risk bands
    st.subheader("Risk scores (next 7/14/30 days)")
    for fc_h, risk_h in bands:
        fig = px.line(
            risk_h, x="datetime", y="risk_score",
            title=f"{state}: Risk score (horizon {risk_h['horizon'].iloc[0]} days)",
            labels={"datetime": COLUMN_LABELS["datetime"], "risk_score": "Risk Score (0–100)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Timeline plot with heatwave shading
    st.subheader("Temperature and heatwave timeline")
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=df_hw["datetime"], y=df_hw["tempmax"], name=COLUMN_LABELS["tempmax"], mode="lines"))
    fig_t.add_trace(go.Scatter(x=df_hw["datetime"], y=df_hw["thr"], name=f"{int(pct*100)}th pct threshold", mode="lines"))
    for _, row in events.iterrows():
        fig_t.add_vrect(x0=row["start"], x1=row["end"], fillcolor="red", opacity=0.15, line_width=0)
    fig_t.update_layout(title=f"{state}: {COLUMN_LABELS['tempmax']} and heatwave periods", height=420)
    st.plotly_chart(fig_t, use_container_width=True)

    # Recommendations
    st.subheader("Actionable recommendations")
    st.markdown("""
- **Early warning:** Trigger advisories when forecasted risk exceeds 70 for ≥3 consecutive days.
- **Cooling access:** Deploy temporary cooling centers and water points in high‑risk LGAs.
- **Work scheduling:** Shift outdoor work to cooler hours during high‑risk windows.
- **Health surveillance:** Mobilize community health workers for heat‑related illness monitoring.
- **Drought resilience:** Prioritize water storage, shade structures, and tree cover in recurrent hotspots.
""")

    # Exports
    st.subheader("Exports")
    # Latest horizon risk & forecast for export (7-day by default)
    fc_latest, risk_latest = bands[0]
    st.download_button("Download events CSV", events.to_csv(index=False).encode("utf-8"), file_name=f"{state.lower()}_heatwave_events.csv")
    st.download_button("Download forecast CSV", fc_latest.to_csv(index=False).encode("utf-8"), file_name=f"{state.lower()}_{target_var}_forecast_{fc_latest['horizon'].iloc[0]}d.csv")
    st.download_button("Download risk CSV", risk_latest.to_csv(index=False).encode("utf-8"), file_name=f"{state.lower()}_{target_var}_risk_{risk_latest['horizon'].iloc[0]}d.csv")

    # HTML summary (user can save as PDF via browser print)
    html = html_summary(state, events, fc_latest, risk_latest)
    st.download_button("Download HTML summary", html.encode("utf-8"), file_name=f"{state.lower()}_heatcast_summary.html")

# -----------------------------
# About
# -----------------------------
elif selected == "About":
    st.header("About HeatCast")
    st.markdown("""
**Program:** Gen Data  
**POD:** HeatCast — Heatwave Risk Analysis and Forecasting in Northeast Nigeria

**Problem Statement:** Rising temperatures in Sokoto, Kebbi, Katsina, Jigawa, and Zamfara threaten communities, yet planners lack localized forecasts to guide proactive interventions.

**Contributions:** Evidence‑based forecasts and recommendations to mitigate heatwave impacts, strengthen drought resilience, and protect vulnerable populations.

**Team:** Members of the POD collaborating across data engineering, modeling, and policy translation.

**Contact:** For collaboration or deployment inquiries, reach out to the POD coordinators.
""")

