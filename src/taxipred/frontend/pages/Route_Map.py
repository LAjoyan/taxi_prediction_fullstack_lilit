import sys
import os
from pathlib import Path
import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent.parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.taxipred.backend.data_processing import build_features
from src.taxipred.utils.constants import USD_TO_SEK, ORS_API_KEY, get_route_data

MODEL_PATH = BASE_DIR / "src" / "taxipred" / "backend" / "random_forest_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


st.set_page_config(page_title="Address Predictor", page_icon="📍")

st.markdown(
    """
    <style>
        .block-container { 
            padding-top: 1rem !important; 
            padding-bottom: 0rem !important; 
        }
        h1 { 
            margin-top: 0rem !important; 
            margin-bottom: 0.5rem !important; 
            font-size: 2rem !important;
        }
        .stAlert { margin-bottom: 1rem !important; }
        iframe { height: 350px !important; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("📍 Address-to-Address Prediction")

st.markdown(
    """
    <div style='background-color: rgba(255, 165, 0, 0.1); padding: 15px; border-left: 5px solid #ffa500; border-radius: 5px; margin-bottom: 20px;'>
        <strong>📏 Service Boundary:</strong> This estimator is designed for city trips. 
        Please ensure your route is within <b>100 km</b>.
    </div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Conditions")
    time_of_day = st.selectbox(
        "Time of Day", ["Morning", "Afternoon", "Evening", "Night"]
    )
    day_of_week = st.selectbox("Day of Week", ["Weekday", "Weekend"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"])

    st.markdown(
        """<br>
            <br>
    <p style='text-align: center;  font-size:20px;'>
    Developed by MLOps student <a href='https://www.linkedin.com/in/lilit-ajoyan-1565b4183/' target='_blank' style='color: #60a5fa; text-decoration: underline;'>
        Lilit Ajoyan 
        </a>
        <br>
        Find the repo on 
    <a href='https://github.com/LAjoyan/taxi_prediction_fullstack_lilit' target='_blank' style='color: #60a5fa; text-decoration: underline;'>
            GitHub
        </a>
            

    """,
        unsafe_allow_html=True,
    )


col_addr1, col_addr2, col_btn = st.columns([2, 2, 1.2])

with col_addr1:
    from_address = st.text_input(
        "From Address",
        placeholder="Enter starting point...",
        label_visibility="collapsed",
    )

with col_addr2:
    to_address = st.text_input(
        "To Address", placeholder="Enter destination...", label_visibility="collapsed"
    )

with col_btn:
    predict_clicked = st.button("Predict Fare", use_container_width=True)

if predict_clicked:
    if not from_address or not to_address:
        st.warning("Please enter both addresses.")

    else:
        try:
            with st.spinner("Calculating ..."):
                route_data = get_route_data(from_address, to_address)

                if route_data["distance_km"] > 100:
                    st.error(
                        "🚨 Distance too far! This model is only for city trips under 100km."
                    )
                else:
                    st.session_state["map_route"] = route_data

                    payload = {
                        "Trip_Distance_km": float(route_data["distance_km"]),
                        "Trip_Duration_Minutes": float(route_data["duration_min"]),
                        "Time_of_Day": time_of_day,
                        "Day_of_Week": day_of_week,
                        "Traffic_Conditions": traffic,
                        "Weather": weather,
                    }

                    if model is not None:
                        X_in = build_features(payload)
                        pred_log = float(model.predict(X_in)[0])
                        pred_price_usd = float(np.expm1(pred_log))
                        pred_price_sek = pred_price_usd * USD_TO_SEK

                        st.session_state["map_prediction"] = {
                            "estimated_price": round(pred_price_sek, 2)
                        }
        except Exception as e:
            st.error(f"Error: {e}")


if "map_route" in st.session_state and "map_prediction" in st.session_state:
    route = st.session_state["map_route"]
    res = st.session_state["map_prediction"]
    st.success(f"Route Found: {route['distance_km']:.2f} km")

    st.markdown(
        f"""
    <div style='background-color:#1e293b; padding:5px; border-radius:10px; text-align:center; border:1px solid #3b82f6;'>
         <h1 style='color:#60a5fa; margin:0;'>{res["estimated_price"]:.2f} SEK</h1>
     </div>
            """,
        unsafe_allow_html=True,
    )
    m = folium.Map(
        location=[
            (route["start_lat"] + route["end_lat"]) / 2,
            (route["start_lon"] + route["end_lon"]) / 2,
        ],
        zoom_start=12,
    )
    folium.PolyLine(route["polyline_latlon"], weight=5).add_to(m)
    st_folium(m, width=700, height=400)
