# app.py
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

from datetime import timedelta
import calendar


# Deep learning & Prophet
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
# Config
# -----------------------------
st.set_page_config(page_title="HeatCast", layout="wide")

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
    "hi": "Heat Index (°C)",
    "solarradiation": "Solar Radiation",
    "uvindex": "Utlta Violet Index",
    "cloudcover": "Cloud Cover",
    "solarenergy": "Solar Energy",
    "conditions": "Weather Condition"
}

USE_COLS = ["name","datetime","tempmax","tempmin","temp","humidity","precip","windspeed","windgust", 
            "solarradiation", "uvindex", "cloudcover", "solarenergy", "conditions"]

# -----------------------------
# Utilities
# -----------------------------
def clean_units(df):
    df = df.copy()
    # Convert Fahrenheit to Celsius
    for col in ["tempmax","tempmin","temp"]:
        if col in df.columns:
            df[col] = (df[col] - 32) * 5.0/9.0
            df[col] = df[col].mask((df[col] < -10) | (df[col] > 55), np.nan)
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(0,100)
    if "precip" in df.columns:
        df["precip"] = df["precip"].clip(lower=0)
    return df

def compute_heat_index(df):
    if not {"tempmax","humidity"}.issubset(df.columns):
        df["hi"] = np.nan
        return df
    T_c = df["tempmax"]
    RH = df["humidity"]
    T_f = T_c*9/5+32
    HI_f = (-42.379 + 2.04901523*T_f + 10.14333127*RH
            -0.22475541*T_f*RH -0.00683783*T_f**2
            -0.05481717*RH**2 +0.00122874*T_f**2*RH
            +0.00085282*T_f*RH**2 -0.00000199*T_f**2*RH**2)
    HI_f = np.where(T_f<80,T_f,HI_f)
    df["hi"] = (HI_f-32)*5/9
    return df

def compute_threshold(df,col="tempmax",pct=0.95):
    return df[col].quantile(pct)

def detect_heatwaves(df,col="tempmax",thr=None,min_consec=3):
    if thr is None: thr = compute_threshold(df,col)
    df["thr"]=thr
    df["hot_day"]=(df[col]>=df["thr"]).astype(int)
    df["exceed"]=(df[col]-df["thr"]).clip(lower=0)
    runs=[]; run_start=None; run_len=0
    for i,row in df.iterrows():
        if row["hot_day"]==1:
            if run_len==0: run_start=row["datetime"]
            run_len+=1
        else:
            if run_len>=min_consec:
                runs.append((run_start,df.loc[i-1,"datetime"],run_len))
            run_len=0
    if run_len>=min_consec:
        runs.append((run_start,df.iloc[-1]["datetime"],run_len))
    df["event_id"]=np.nan; eid=0
    for start,end,length in runs:
        mask=(df["datetime"]>=start)&(df["datetime"]<=end)
        df.loc[mask,"event_id"]=eid; eid+=1
    events=(df.dropna(subset=["event_id"])
              .groupby("event_id")
              .agg(start=("datetime","min"),end=("datetime","max"),
                   duration=("datetime","count"),
                   peak_intensity=("exceed","max"),
                   mean_intensity=("exceed","mean"))
              .reset_index())
    return df,events

def narrative(df,col="tempmax",thr=None):
    latest=df.tail(30)
    mean_val=latest[col].mean(); max_val=latest[col].max()
    if thr is None: thr=compute_threshold(df,col)
    hot_share=(latest[col]>=thr).mean()
    obs=f"Recent {COLUMN_LABELS.get(col,col)} averages ~{mean_val:.1f}°C, peaks {max_val:.1f}°C. {hot_share*100:.0f}% exceeded threshold."
    imp="Sustained highs elevate heat stress, raising risks for vulnerable groups and outdoor workers."
    rec="Issue advisories, expand cooling/water access, adjust work schedules, and monitor health impacts."
    return obs,imp,rec


# -----------------------------
# Forecast helpers
# -----------------------------
def forecast_arima(df,target="tempmax",horizon=7):
    series=df.set_index("datetime")[target].dropna()
    model=ARIMA(series,order=(2,1,2)).fit()
    fc=model.get_forecast(steps=horizon)
    fdf=pd.DataFrame({"datetime":pd.date_range(series.index.max()+timedelta(days=1),periods=horizon,freq="D"),
                      target:fc.predicted_mean.values})
    return fdf

def forecast_sarima(df,target="tempmax",horizon=7):
    series=df.set_index("datetime")[target].dropna()
    model=SARIMAX(series,order=(2,1,2),seasonal_order=(1,1,1,7)).fit(disp=False)
    fc=model.get_forecast(steps=horizon)
    fdf=pd.DataFrame({"datetime":pd.date_range(series.index.max()+timedelta(days=1),periods=horizon,freq="D"),
                      target:fc.predicted_mean.values})
    return fdf

def forecast_ml(df,target="tempmax",horizon=7,model_name="XGB"):
    series=df.set_index("datetime")[target].dropna()
    X=np.arange(len(series)).reshape(-1,1); y=series.values
    if model_name=="XGB":
        model=XGBRegressor(n_estimators=200,random_state=42)
    elif model_name=="LGBM":
        model=LGBMRegressor(n_estimators=200,random_state=42)
    else:
        model=GradientBoostingRegressor(random_state=42)
    model.fit(X,y)
    future=np.arange(len(series),len(series)+horizon).reshape(-1,1)
    preds=model.predict(future)
    fdf=pd.DataFrame({"datetime":pd.date_range(series.index.max()+timedelta(days=1),periods=horizon,freq="D"),
                      target:preds})
    return fdf

def forecast_prophet(df: pd.DataFrame, target="tempmax", horizon=7):
    if Prophet is None:
        raise RuntimeError("Prophet not available in environment.")
    d = df[["datetime", target]].dropna().rename(columns={"datetime":"ds", target:"y"})
    m = Prophet(seasonality_mode="additive", yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(d)
    future = m.make_future_dataframe(periods=horizon, freq="D")
    fc = m.predict(future).tail(horizon)[["ds","yhat"]].rename(columns={"ds":"datetime","yhat":target})
    return fc, m

# -----------------------------
# Navigation
# -----------------------------
selected=option_menu(
    menu_title="HeatCast",
    options=["Home","Climate Insights"],
    icons=["house","bar-chart"],
    orientation="horizontal"
)

# -----------------------------
# Home Page
# -----------------------------
if selected=="Home":
    st.title("HeatCast: Heatwave Risk Analysis and Forecasting")
    st.markdown("""
Welcome to **HeatCast**, part of the **Gen Data program**.  
This app empowers communities and policymakers by analyzing heatwave risks, forecasting future conditions, and providing actionable recommendations.

### Contributors
- Gen Data POD members
- Climate scientists, data engineers, and policy experts

Use the **Climate Insights** page to upload your weather data and explore analysis, forecasts, and risk dashboards.
""")

# -----------------------------
# Climate Insights Page
# -----------------------------
elif selected=="Climate Insights":
    st.sidebar.title("Upload and Settings")
    st.sidebar.markdown(""" Upload a CSV with columns such as:
    1. `name`: Station Name
    2. `datetime`: Date/Time
    3. `tempmax`: Maximum Temperature (°C)
    4. `tempmin`: Minimum Temperature (°C)
    5. `temp`: Average Temperature (°C)
    6. `humidity`: Relative Humidity (%)
    7. `precip`: Precipitation (mm)
    8. `windspeed`: Wind Speed (km/h)
    9. `windgust`: Wind Gust (km/h)
    10. `solarradiation`: Solar Radiation
    11. `uvindex`: Utlta Violet Index
    12. `cloudcover`: Cloud Cover
    13. `solarenergy`: Solar Energy
    14. `conditions`: Weather Condition """) # name, datetime, tempmax, tempmin, temp, humidity, precip, windspeed, windgust")
    uploaded=st.sidebar.file_uploader("Upload CSV",type=["csv"])
    if uploaded:
        df=pd.read_csv(uploaded)
        df=df[USE_COLS]
        df["datetime"]=pd.to_datetime(df["datetime"],errors="coerce")
        df=clean_units(df)
        df=compute_heat_index(df)
        st.success("Data uploaded and processed!")
    
        # -----------------------------
        # Analysis Section
        # -----------------------------
        with st.expander("Analysis"):
            required_cols = ["solarradiation", "cloudcover", "uvindex"]
            st.subheader("Heatwave Weather Variable Analysis")

            analysis_type = st.radio("Select analysis type", 
                                    ["Single Variable Analysis",
                                    "Multiple Variable Analysis",
                                    "Heat Risk Analysis"])

            # -----------------------------
            # Single Variable Analyses
            # -----------------------------
            if analysis_type == "Single Variable Analysis":
                single_option = st.selectbox("Choose analysis", 
                                            ["Diurnal Temperature Range (DTR)",
                                            "Distribution of Apparent Temperature"])

                if single_option == "Diurnal Temperature Range (DTR)":
                    if not set(USE_COLS).issubset(df.columns):
                        st.warning(f"This analysis cannot be done as the weather variable selected in not part of the '{USE_COLS}' in your dataset.")
                    else:
                        df["DTR"] = df["tempmax"] - df["tempmin"]
                        fig = px.line(df, x="datetime", y="DTR",
                                    title="Diurnal Temperature Range (°C)",
                                    labels={"datetime":COLUMN_LABELS["datetime"],"DTR":"DTR (°C)"})
                        st.plotly_chart(fig,use_container_width=True)

                        st.markdown("**Observation:** DTR values show how much temperatures swing between day and night. Narrow ranges indicate persistent heat stress, while wider ranges suggest nighttime relief.")
                        st.markdown("**Implications:** Persistently low DTR means communities don’t cool off at night, increasing cumulative heat stress and health risks.")
                        st.markdown("**Recommendations:** Promote nighttime cooling strategies (ventilation, shaded sleeping areas) in regions with low DTR.")

                elif single_option == "Distribution of Apparent Temperature":
                    if not set(USE_COLS).issubset(df.columns):
                        st.warning(f"This analysis cannot be done as the weather variable selected in not part of the '{USE_COLS}' in your dataset.")
                    else:
                        fig = px.histogram(df, x="hi", nbins=30,
                                        title="Distribution of Apparent Temperature (Heat Index)",
                                        labels={"hi":COLUMN_LABELS["hi"]})
                        st.plotly_chart(fig,use_container_width=True)

                        st.markdown("**Observation:** The distribution shows how often apparent temperatures exceed physiological thresholds (e.g., >40 °C).")
                        st.markdown("**Implications:** Apparent temperature combines humidity and heat, so high values indicate dangerous conditions even when raw temperature is moderate.")
                        st.markdown("**Recommendations:** Base advisories on apparent temperature, not just raw temperature, to better capture human risk.")

            # -----------------------------
            # Heat Risk Analysis
            # -----------------------------
            elif analysis_type == "Heat Risk Analysis":
                if not set(USE_COLS).issubset(df.columns):
                        st.warning(f"This analysis cannot be done as the weather variable selected in not part of the '{USE_COLS}' in your dataset.")
                else:
                    # Thresholds
                    def compute_heatwave_thresholds(df, variable='tempmax'):
                        thresholds = {
                            "Advisory": df[variable].quantile(0.90),
                            "Warning": df[variable].quantile(0.95),
                            "Emergency": df[variable].quantile(0.975)
                        }
                        return thresholds

                    target = st.selectbox(
                        "Select Weather Variable",
                        ["tempmax","tempmin","temp","humidity","uvindex","solarradiation","precip"],
                        key='selectbox_heatrisk',
                        format_func=lambda x: COLUMN_LABELS.get(x,x)
                    )

                    thresholds = compute_heatwave_thresholds(df, variable=target)

                    def detect_heatwave_events(df, variable, threshold, min_days=3):
                        df = df.copy()
                        df['above_threshold'] = df[variable] >= threshold
                        df['heatwave_group'] = (df['above_threshold'] != df['above_threshold'].shift()).cumsum()
                        heatwave_days = (
                            df[df['above_threshold']]
                            .groupby('heatwave_group')
                            .filter(lambda x: len(x) >= min_days)
                        )
                        return heatwave_days

                    def classify_heat_alert(value, thresholds):
                        if value >= thresholds['Emergency']:
                            return "Heat Emergency"
                        elif value >= thresholds['Warning']:
                            return "Heat Warning"
                        elif value >= thresholds['Advisory']:
                            return "Heat Advisory"
                        else:
                            return "Normal"

                    df['heat_alert'] = df[target].apply(lambda x: classify_heat_alert(x, thresholds))

                    fig1 = px.line(
                        df,
                        x='datetime',
                        y=target,
                        color='heat_alert',
                        title=f"Heatwave Alerts – {COLUMN_LABELS.get(target,target)}",
                        color_discrete_map={
                            "Normal": "green",
                            "Heat Advisory": "orange",
                            "Heat Warning": "red",
                            "Heat Emergency": "darkred"
                        }
                    )

                    for label, value in thresholds.items():
                        fig1.add_hline(y=value, line_dash="dash")

                    st.plotly_chart(fig1, use_container_width=True)

                    # Narration
                    latest_alert = df.iloc[-1]['heat_alert']
                    st.subheader("Observation")
                    st.write(f"The most recent conditions indicate a **{latest_alert}** level based on localized heat thresholds.")

                    st.subheader("Implication")
                    st.write("Elevated heat levels increase the risk of heat exhaustion, dehydration, and reduced productivity, especially among outdoor workers and vulnerable populations.")

                    st.subheader("Recommendation")
                    if latest_alert in ["Heat Warning", "Heat Emergency"]:
                        st.write("Immediate activation of heat action plans is advised, including public alerts, cooling shelters, and adjusted work schedules.")
                    else:
                        st.write("Continued monitoring is recommended, with preparedness measures in place ahead of peak heat periods.")

                    st.markdown("""**Note**:  
                    *We adopt a percentile-based heatwave definition to ensure thresholds are locally adaptive and scientifically consistent with WMO and IPCC methodologies. The 90th, 95th, and 97.5th percentiles represent increasing levels of extremity and are widely used in heatwave literature.*""")

                    # -----------------------------
                    # Derived Heat Risk Index (HRI)
                    # -----------------------------
                    def normalize_columns(df, cols):
                        scaler = MinMaxScaler()
                        df_scaled = df.copy()
                        df_scaled[cols] = scaler.fit_transform(df[cols])
                        return df_scaled

                    def compute_heat_risk_index(df):
                        cols = ['tempmax','humidity','windspeed']
                        # Only include optional columns if they exist
                        if 'solarradiation' in df.columns:
                            cols.append('solarradiation')
                        if 'feelslikemax' in df.columns:
                            cols.append('feelslikemax')

                        df_clean = df[cols].dropna()
                        df_norm = normalize_columns(df_clean, cols)

                        # Weighted sum (skip missing weights gracefully)
                        df['HRI'] = 0
                        if 'tempmax' in df_norm: df['HRI'] += 0.30 * df_norm['tempmax']
                        if 'feelslikemax' in df_norm: df['HRI'] += 0.25 * df_norm['feelslikemax']
                        if 'humidity' in df_norm: df['HRI'] += 0.20 * df_norm['humidity']
                        if 'solarradiation' in df_norm: df['HRI'] += 0.15 * df_norm['solarradiation']
                        if 'windspeed' in df_norm: df['HRI'] -= 0.10 * df_norm['windspeed']

                        return df


                    def classify_hri(hri):
                        if hri >= 0.75:
                            return "Extreme Risk"
                        elif hri >= 0.55:
                            return "High Risk"
                        elif hri >= 0.35:
                            return "Moderate Risk"
                        else:
                            return "Low Risk"

                    df = compute_heat_risk_index(df)
                    df['HRI_Level'] = df['HRI'].apply(classify_hri)

                    fig2 = px.line(
                        df,
                        x='datetime',
                        y='HRI',
                        color='HRI_Level',
                        title="Heat Risk Index (HRI)",
                        color_discrete_map={
                            "Low Risk": "green",
                            "Moderate Risk": "yellow",
                            "High Risk": "orange",
                            "Extreme Risk": "red"
                        }
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    current_hri = df.iloc[-1]['HRI_Level']
                    st.subheader("Observation")
                    st.write(f"The current Heat Risk Index indicates **{current_hri}** conditions.")

                    st.subheader("Implication")
                    st.write("High or extreme heat risk conditions increase susceptibility to heat-related illnesses, reduce labor efficiency, and elevate strain on healthcare and energy systems.")

                    st.subheader("Recommendation")
                    st.write("Risk-informed planning should include heat-resilient infrastructure, targeted public advisories, and integration of HRI into early warning systems.")

                    st.markdown("""**Note**:  
                    *The HRI weights reflect the relative contribution of heat intensity, physiological stress and environmental amplification, consistent with established heat stress frameworks while remaining interpretable and data-efficient.*  
                    *HRI thresholds were defined on a normalized scale to reflect progressive risk escalation, with the intent that they remain adaptable and subject to calibration as additional impact data becomes available.*""")


            # -----------------------------
            # Multiple Variable Analyses
            # -----------------------------
            else:
                multi_option = st.selectbox("Choose analysis", 
                                            ["Humidity vs Temperature (Danger Zone)",
                                            "Solar Radiation vs Max Temp",
                                            "Cloud Cover and Heat Retention",
                                            "3D Heat Vulnerability Matrix",
                                            "Rainfall Deficit and Heat Correlation",
                                            "Radiation and Energy Exposure",
                                            "Monthly Heat Intensity Heatmap"])
                

                if multi_option == "Humidity vs Temperature (Danger Zone)":
                    if not set(USE_COLS).issubset(df.columns):
                        st.warning(f"This analysis cannot be done as the weather variable selected in not part of the '{USE_COLS}' in your dataset.")
                    else:
                        fig = px.scatter(df, x="humidity", y="tempmax", color="hi",
                                        title="Humidity vs Max Temperature (Danger Zone)",
                                        labels={"humidity":COLUMN_LABELS["humidity"],"tempmax":COLUMN_LABELS["tempmax"],"hi":COLUMN_LABELS["hi"]})
                        st.plotly_chart(fig,use_container_width=True)

                        st.markdown("**Observation:** The scatter reveals combinations where high humidity amplifies heat stress — the 'danger zone'.")
                        st.markdown("**Implications:** Outdoor workers face extreme risk when both humidity and temperature are high, as sweat evaporation is impaired.")
                        st.markdown("**Recommendations:** Issue targeted warnings for high humidity + high temperature days; adjust work/rest cycles accordingly.")

                # Motnthly Heat Intensity
                elif multi_option == "Monthly Heat Intensity Heatmap":
                    if not set(USE_COLS).issubset(df.columns):
                        st.warning(f"This analysis cannot be done as the weather variable selected in not part of the '{USE_COLS}' in your dataset.")
                    else:
                        # Ensure we have month and year columns
                        df["Year"] = df["datetime"].dt.year
                        df["Month_Name"] = df["datetime"].dt.month_name()
                        df["Month_Num"] = df["datetime"].dt.month

                        # Pivot table: average max temp by month/year
                        heatmap_data = df.pivot_table(values="tempmax", 
                                                    index="Month_Num", 
                                                    columns="Year", 
                                                    aggfunc="mean")

                        # Reindex months to chronological order
                        month_order = list(range(1,13))
                        heatmap_data = heatmap_data.reindex(month_order)

                        # Replace index with month names
                        heatmap_data.index = [calendar.month_name[m] for m in heatmap_data.index]

                        # Plotly heatmap
                        fig = px.imshow(
                            heatmap_data,
                            labels=dict(x="Year", y="Month", color="Average Max Temperature (°C)"),
                            x=heatmap_data.columns,
                            y=heatmap_data.index,
                            color_continuous_scale="RdYlBu_r",
                            aspect="auto",
                            title="Monthly Heat Intensity across Years"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Narratives
                        st.subheader("Observation")
                        st.write(
                            "The heatmap reveals seasonal cycles of maximum temperature across years. "
                            "Peak intensities consistently occur in the pre-rainy season months (March–May), "
                            "while cooler conditions are observed during the rainy season (July–September). "
                            "Interannual variability is visible, with some years showing prolonged high-intensity months."
                        )

                        st.subheader("Implication")
                        st.write(
                            "This pattern indicates that communities face predictable seasonal heat stress windows. "
                            "Periods of prolonged high temperatures coincide with agricultural planting seasons, "
                            "increasing vulnerability to crop failure and water scarcity. "
                            "Interannual variability suggests climate change is amplifying extremes, "
                            "making planning more complex."
                        )

                        st.subheader("Recommendation")
                        st.write(
                            "Policy makers should integrate seasonal heat forecasts into agricultural calendars, "
                            "infrastructure planning, and public health advisories. "
                            "Cooling interventions and water storage should be prioritized ahead of peak heat months. "
                            "Long-term adaptation strategies must account for increasing interannual variability."
                        )

                elif multi_option == "Solar Radiation vs Max Temp":
                    if not set(USE_COLS).issubset(df.columns):
                        st.warning(f"This analysis cannot be done as the weather variable selected in not part of the '{USE_COLS}' in your dataset.")
                    else:
                        fig = px.scatter(df, x="solarradiation", y="tempmax",
                                        title="Solar Radiation vs Maximum Temperature",
                                        labels={"solarradiation":"Solar Radiation (W/m²)","tempmax":COLUMN_LABELS["tempmax"]})
                        st.plotly_chart(fig,use_container_width=True)

                        st.markdown("**Observation:** Strong correlation indicates solar load drives peak temperatures.")
                        st.markdown("**Implications:** High solar radiation days intensify heatwaves, especially in urban areas with low albedo.")
                        st.markdown("**Recommendations:** Promote reflective roofing and shading to reduce solar heat absorption.")

                elif multi_option == "Cloud Cover and Heat Retention":
                    if not set(USE_COLS).issubset(df.columns):
                        st.warning(f"This analysis cannot be done as the weather variable selected in not part of the '{USE_COLS}' in your dataset.")
                    else:
                        df["cloud_bin"] = pd.cut(df["cloudcover"], bins=[0,25,50,75,100])
                        fig = px.box(df, x="cloud_bin", y="tempmax",
                                    title="Cloud Cover vs Maximum Temperature",
                                    labels={"cloud_bin":"Cloud Cover (%)","tempmax":COLUMN_LABELS["tempmax"]})
                        st.plotly_chart(fig,use_container_width=True)

                        st.markdown("**Observation:** Low cloud cover correlates with higher daytime peaks; high cloud cover traps heat at night.")
                        st.markdown("**Implications:** Cloud dynamics influence both daytime extremes and nighttime cooling.")
                        st.markdown("**Recommendations:** Integrate cloud cover forecasts into heatwave early warning systems.")

                elif multi_option == "3D Heat Vulnerability Matrix":
                    if not set(USE_COLS).issubset(df.columns):
                        st.warning(f"This analysis cannot be done as the weather variable selected in not part of the '{USE_COLS}' in your dataset.")
                    else:
                        fig = px.scatter_3d(df, x="temp", y="humidity", z="uvindex", color="tempmax",
                                            title="3D Heat Vulnerability Matrix",
                                            labels={"temp":COLUMN_LABELS["temp"],"humidity":COLUMN_LABELS["humidity"],"uvindex":"UV Index","tempmax":COLUMN_LABELS["tempmax"]})
                        st.plotly_chart(fig,use_container_width=True)

                        st.markdown("**Observation:** Reveals compound exposure zones where heat, humidity, and UV all peak.")
                        st.markdown("**Implications:** Vulnerability is multidimensional — high UV plus high heat and humidity increases dehydration and skin cancer risk.")
                        st.markdown("**Recommendations:** Combine UV advisories with heat warnings; promote sunscreen and shaded outdoor work.")

                elif multi_option == "Rainfall Deficit and Heat Correlation":
                    if not set(USE_COLS).issubset(df.columns):
                        st.warning(f"This analysis cannot be done as the weather variable selected in not part of the '{USE_COLS}' in your dataset.")
                    else:
                        df["precip_roll7"] = df["precip"].rolling(7).mean()
                        fig = px.line(df, x="datetime", y=["precip_roll7","tempmax"],
                                    title="Rainfall Deficit & Heat Correlation",
                                    labels={"precip_roll7":"7-day Rolling Precipitation (mm)","tempmax":COLUMN_LABELS["tempmax"]})
                        st.plotly_chart(fig,use_container_width=True)

                        st.markdown("**Observation:** Heatwaves often coincide with rainfall deficits, worsening drought stress.")
                        st.markdown("**Implications:** Lack of rain reduces soil moisture, amplifying heat extremes and agricultural losses.")
                        st.markdown("**Recommendations:** Couple heatwave alerts with drought preparedness (water storage, irrigation scheduling).")

                elif multi_option == "Radiation and Energy Exposure":
                    if not set(USE_COLS).issubset(df.columns):
                        st.warning(f"This analysis cannot be done as the weather variable selected in not part of the variables to use ({USE_COLS}) in your dataset.")
                    else:
                        fig = px.scatter(df, x="uvindex", y="solarenergy", size="temp", color="conditions",
                                        title="Radiation & Energy Exposure",
                                        labels={"uvindex":"UV Index","solarenergy":"Solar Energy (MJ/m²)","temp":COLUMN_LABELS["temp"],"conditions":"Weather Conditions"})
                        st.plotly_chart(fig,use_container_width=True)

                        st.markdown("**Observation:** Shows how solar energy and UV exposure combine with temperature to drive human and ecological stress.")
                        st.markdown("**Implications:** High radiation days increase energy demand (cooling) and health risks.")
                        st.markdown("**Recommendations:** Promote solar energy harvesting on high radiation days to offset cooling demand.")

        # -----------------------------
        # Forecast Section
        # -----------------------------
        with st.expander("Forecast"):
            st.subheader("Temperature Forecasting")

            model_choice = st.selectbox("Model", ["SARIMA","ARIMA","XGBoost","LightGBM","LSTM","Prophet"])
            horizon = st.select_slider("Forecast horizon (days)", options=[7,14,30], value=7)
            target_var = st.selectbox("Target variable", ["tempmax","temp","hi"],
                                      format_func=lambda x: COLUMN_LABELS.get(x,x))

            if st.button("Run Forecast"):
                with st.spinner("Forecast in progress… some models may take longer."):
                    try:
                        if model_choice=="SARIMA":
                            fc = forecast_sarima(df,target=target_var,horizon=horizon)
                        elif model_choice=="ARIMA":
                            fc = forecast_arima(df,target=target_var,horizon=horizon)
                        elif model_choice in ["XGBoost","LightGBM"]:
                            fc = forecast_ml(df,target=target_var,horizon=horizon,
                                             model_name="XGB" if model_choice=="XGBoost" else "LGBM")
                        elif model_choice=="LSTM":
                            fc = forecast_ml(df,target=target_var,horizon=horizon,model_name="GBR") # placeholder
                        elif model_choice=="Prophet" and Prophet is not None:
                            fc,_ = forecast_prophet(df,target=target_var,horizon=horizon)
                        else:
                            st.error("Model not available in this environment.")
                            fc=None

                        if fc is not None:
                            hist_tail = df[["datetime",target_var]].tail(120)
                            fc_all = pd.concat([hist_tail,fc],ignore_index=True)
                            fig = px.line(fc_all,x="datetime",y=target_var,
                                          color=(fc_all["datetime"]>df["datetime"].max()).map({True:"Forecast",False:"History"}),
                                          title=f"{COLUMN_LABELS.get(target_var,target_var)} forecast ({model_choice}, {horizon} days)",
                                          labels={"datetime":COLUMN_LABELS["datetime"], target_var:COLUMN_LABELS.get(target_var,target_var)})
                            st.plotly_chart(fig,use_container_width=True)

                            obs,imp,rec = narrative(pd.concat([df,fc_all]),col=target_var)
                            st.markdown(f"**Observation:** {obs}")
                            st.markdown(f"**Implications:** {imp}")
                            st.markdown(f"**Recommendations:** {rec}")

                            st.download_button("Download forecast CSV", fc.to_csv(index=False).encode("utf-8"),
                                               file_name=f"{target_var}_forecast_{horizon}d.csv")
                    except Exception as e:
                        st.error(f"Forecast error: {e}")

        # -----------------------------
        # Heat Risk Section
        # -----------------------------
        with st.expander("Heat Risk"):
            st.subheader("Heatwave Risk Dashboard")

            thr = compute_threshold(df,"tempmax",pct=0.95)
            df_hw,events = detect_heatwaves(df,"tempmax",thr=thr,min_consec=3)
            st.dataframe(events)

            fc7 = forecast_ml(df,target="tempmax",horizon=7)
            risk7 = fc7.copy()
            risk7["thr"]=thr
            risk7["risk_score"] = 100*((risk7["tempmax"]-thr).clip(lower=0)/(thr+1e-6))

            fig_risk = px.line(risk7,x="datetime",y="risk_score",
                               title="Heatwave Risk Score (next 7 days)",
                               labels={"datetime":COLUMN_LABELS["datetime"],"risk_score":"Risk Score (0–100)"})
            st.plotly_chart(fig_risk,use_container_width=True)

            st.markdown("""
            **Recommendations:**
            - Trigger advisories when risk exceeds 70 for ≥3 consecutive days.
            - Deploy cooling centers and water points in high‑risk areas.
            - Adjust outdoor work schedules to cooler hours.
            - Mobilize health workers for heat‑related illness monitoring.
            """)

            st.download_button("Download events CSV", events.to_csv(index=False).encode("utf-8"), file_name="heatwave_events.csv")
            st.download_button("Download risk CSV", risk7.to_csv(index=False).encode("utf-8"), file_name="heat_risk.csv")
    else:
        st.markdown("""
                ### Welcome to HeatCast App.
                **Upload a weather data in CSV format to start analysis and forecast. Check the sidebar for instruction!**

                    """)
