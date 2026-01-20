# Core
import pandas as pd
import numpy as np
from datetime import timedelta
import streamlit as st
from streamlit_option_menu import option_menu

# Visualization
import plotly.express as px

# Scaling & Metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet
from prophet import Prophet

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set page config
st.set_page_config(
    page_title="HeatCast – Heatwave Risk Analysis",
    layout="wide"
)


# Navigation menu
selected = option_menu(
    menu_title="HeatCast – Heatwave Risk Analysis", 
    options=["Overview", "Analysis", "Train Model", "Make Prediction", "About"],
    icons=["house", "bar-chart", "cpu", "check-circle", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

@st.cache_data
def load_data(location):
    file_map = {
        "Sokoto": "Sokoto Climate Data - 2010-Nov-2025.csv",
        #"Kebbi": "data/kebbi_weather.csv",
        #"Katsina": "data/katsina_weather.csv",
        "Jigawa": "Jigawa Climate Data - 2010-Nov_2025.csv",
        #"Zamfara": "data/zamfara_weather.csv"
    }
    
    df = pd.read_csv(file_map[location])
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


# Overview Page
if selected == "Overview":
    st.title("HeatCast: Heatwave Risk Analysis and Forecasting")
    st.subheader("Localized Climate Intelligence for Policy and Planning in Northern Nigeria")

    #st.subheader("Data Preview")
    #df = load_data("Sokoto")

    #df.head()

    st.sidebar.header("User Controls")

    location = st.sidebar.selectbox(
        "Select Location",
        ["Sokoto", "Kebbi", "Katsina", "Jigawa", "Zamfara"]
    )

    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["Single Variable Analysis", "Multivariable Analysis", "Forecasting"]
    )

    df = load_data(location)

    # Analysis
    if analysis_type == "Single Variable Analysis":

        st.header(f"Single Variable Analysis – {location}")

        variable = st.selectbox(
            "Select Weather Variable",
            [
                "tempmax",
                "tempmin",
                "temp",
                "feelslikemax",
                "humidity",
                "uvindex",
                "solarradiation",
                "precip"
            ],
            key='selectbox_1'
        )

        # Long-Term Trend
        fig = px.line(
                df,
                x="datetime",
                y=variable,
                title=f"{variable.upper()} Trend Over Time ({location})",
                labels={
                    "datetime": "Date",
                    variable: variable.upper()
                }
            )

        st.plotly_chart(fig, use_container_width=True)
    
        # Observation, Implication & Recommendation
        mean_value = df[variable].mean()
        recent_mean = df[df['datetime'] >= "2020-01-01"][variable].mean()

        st.subheader("Observation")
        st.markdown(
            f"The average **{variable}** over the entire period is **{mean_value:.2f}**, with recent years (post-2020) averaging **{recent_mean:.2f}**, indicating an upward pressure in recent climatic conditions.")

        st.subheader("Implication")
        st.write(
            "Sustained increases in this variable elevate heat stress levels, "
            "pose risks to human health, reduce outdoor productivity, "
            "and intensify pressure on water and energy systems."
        )

        st.subheader("Recommendation")
        st.write(
            "Policymakers should integrate heat-adaptive infrastructure planning, "
            "strengthen early warning systems, and prioritize vulnerable populations "
            "during peak heat periods."
        )


    if analysis_type == "Multivariable Analysis":

        st.header(f"Multivariable Heat Risk Analysis – {location}")

        x_var = st.selectbox("X Variable", ["humidity", "windspeed", "solarradiation"])
        y_var = st.selectbox("Y Variable", ["tempmax", "feelslikemax"])

        fig = px.scatter(
            df,
            x=x_var,
            y=y_var,
            trendline="ols",
            title=f"{y_var.upper()} vs {x_var.upper()} ({location})"
        )

        st.plotly_chart(fig, use_container_width=True)

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
                [
                    "tempmax",
                    "tempmin",
                    "temp",
                    "feelslikemax",
                    "humidity",
                    "uvindex",
                    "solarradiation",
                    "precip"
                ],
                key='selectbox_2'
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
        
        df['heat_alert'] = df[target].apply(
        lambda x: classify_heat_alert(x, thresholds))

        fig1 = px.line(
            df,
            x='datetime',
            y=target,
            color='heat_alert',
            title=f"Heatwave Alerts – {location}",
            color_discrete_map={
                "Normal": "green",
                "Heat Advisory": "orange",
                "Heat Warning": "red",
                "Heat Emergency": "darkred"
            }
        )

        for label, value in thresholds.items():
            fig1.add_hline(
                y=value,
                line_dash="dash",
                #annotation_text=f"{label} Threshold"
            )

        st.plotly_chart(fig1, use_container_width=True)

        # Narration
        latest_alert = df.iloc[-1]['heat_alert']

        st.subheader("Observation")
        st.write(f"The most recent conditions indicate a **{latest_alert}** level based on localized heat thresholds.")

        st.subheader("Implication")
        st.write(
            "Elevated heat levels increase the risk of heat exhaustion, dehydration, "
            "and reduced productivity, especially among outdoor workers and vulnerable populations."
        )

        st.subheader("Recommendation")
        if latest_alert in ["Heat Warning", "Heat Emergency"]:
            st.write(
                "Immediate activation of heat action plans is advised, including public alerts, "
                "cooling shelters, and adjusted work schedules."
            )
        else:
            st.write(
                "Continued monitoring is recommended, with preparedness measures in place "
                "ahead of peak heat periods."
            )
        
        st.markdown("""**Note**:
                    *We adopt a percentile-based heatwave definition to ensure thresholds are locally adaptive and scientifically consistent with WMO and IPCC methodologies. The 90th, 95th, and 97.5th percentiles represent increasing levels of extremity and are widely used in heatwave literature.*""", width = 'stretch')

        # DERIVED HEAT RISK INDEX (HRI)
        def normalize_columns(df, cols):
            scaler = MinMaxScaler()
            df_scaled = df.copy()
            df_scaled[cols] = scaler.fit_transform(df[cols])
            return df_scaled
        
        def compute_heat_risk_index(df):
            cols = [
                'tempmax',
                'feelslikemax',
                'humidity',
                'solarradiation',
                'windspeed'
            ]

            df_clean = df[cols].dropna()
            df_norm = normalize_columns(df_clean, cols)

            df['HRI'] = (
                0.30 * df_norm['tempmax'] +
                0.25 * df_norm['feelslikemax'] +
                0.20 * df_norm['humidity'] +
                0.15 * df_norm['solarradiation'] -
                0.10 * df_norm['windspeed']
            )

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
            title=f"Heat Risk Index (HRI) – {location}",
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
        st.write(
            "High or extreme heat risk conditions increase susceptibility to heat-related illnesses, "
            "reduce labor efficiency, and elevate strain on healthcare and energy systems."
        )

        st.subheader("Recommendation")
        st.write(
            "Risk-informed planning should include heat-resilient infrastructure, "
            "targeted public advisories, and integration of HRI into early warning systems."
        )

        st.markdown("""**Note**:
                *The HRI weights reflect the relative contribution of heat intensity, physiological stress and environmental amplification, consistent with established heat stress frameworks while remaining interpretable and data-efficient.*
                *HRI thresholds were defined on a normalized scale to reflect progressive risk escalation, with the intent that they remain adaptable and subject to calibration as additional impact data becomes available.*
            """, width = 'stretch')
    


