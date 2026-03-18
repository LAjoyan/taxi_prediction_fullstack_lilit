import os
import sys

sys.path.append(os.getcwd())
from pathlib import Path
import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
from src.taxipred.backend.data_processing import build_features
from src.taxipred.utils.constants import USD_TO_SEK, get_route_data

if "ORS_API_KEY" in st.secrets:
    os.environ["ORS_API_KEY"] = st.secrets["ORS_API_KEY"]

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent.parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

st.set_page_config(page_title="Address Predictor", page_icon="📍")


st.markdown(
    """
    <style>
        /* General layout spacing */
        .block-container { padding-top: 2rem !important; padding-bottom: 0rem !important; }
        h1 { margin-top: 0rem !important; margin-bottom: 0.5rem !important; font-size: 2rem !important; }
        .stAlert { margin-bottom: 1rem !important; }
        
        /* Force the map's iframe to a specific height */
        iframe { height: 400px !important; }

        /* The "Magic" selector to round the map edges */
        [data-testid="stVerticalBlockBorderWrapper"]:has(iframe) {
            border-radius: 10px !important;
            overflow: hidden !important;
            border: 2px solid #3b82f6 !important;
            margin-top: 15px;
        }
        /* Ensures the internal iframe also respects the rounding */
        iframe {
            border-radius: 10px !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

MODEL_PATH = BASE_DIR / "src" / "taxipred" / "backend" / "random_forest_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


with st.sidebar:
    st.header("Conditions")
    time_of_day = st.selectbox(
        "Time of Day", ["Morning", "Afternoon", "Evening", "Night"]
    )
    day_of_week = st.selectbox("Day of Week", ["Weekday", "Weekend"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"])

    st.write("")  # Small spacer
    if st.button("Reset Route", use_container_width=True):
        st.session_state["input_from"] = ""
        st.session_state["input_to"] = ""
        st.session_state.pop("map_route", None)
        st.session_state.pop("map_prediction", None)
        st.rerun()

    st.markdown("---")
    st.markdown(
       """
    <div style='text-align: center; font-size:14px; color: var(--text-color); opacity: 0.7;'>
        Developed by <span style='font-weight: 800; color: var(--text-color);'>Lilit Ajoyan</span><br>
        <a href='https://www.linkedin.com/in/lilit-ajoyan-1565b4183/' target='_blank' style='color: #60a5fa;'>LinkedIn</a> | 
        <a href='https://github.com/LAjoyan/taxi_prediction_fullstack_lilit' target='_blank' style='color: #60a5fa;'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)


st.title("📍 Address-to-Address Prediction")

st.caption("💡 Tip: Change sidebar conditions to see how the price fluctuates!")


st.markdown(
    """
    <div style='background-color: rgba(255, 165, 0, 0.1); padding: 15px; border-left: 5px solid #ffa500; border-radius: 5px; margin-bottom: 20px;'>
        <strong>📏 Service Boundary:</strong> This estimator is designed for city trips. 
        Please ensure your route is within <b>100 km</b>.
    </div>
""",
    unsafe_allow_html=True,
)


col_addr1, col_addr2, col_btn = st.columns([2, 2, 1.2])

with col_addr1:
    from_address = st.text_input(
        "From Address",
        placeholder="Enter starting point...",
        label_visibility="collapsed",
        key="input_from"
    )

with col_addr2:
    to_address = st.text_input(
        "To Address", 
        placeholder="Enter destination...", 
        label_visibility="collapsed",
        key="input_to"
    )

with col_btn:
    if st.button("Predict Fare", use_container_width=True):
        if not from_address or not to_address:
            st.warning("Please enter both addresses.")
        try:
            with st.spinner("Calculating ..."):
                route_data = get_route_data(from_address, to_address)

                if route_data["distance_km"] > 100:
                    st.error(
                        "🚨 Distance too far! This model is only for city trips under 100km."
                    )
                else:
                    st.session_state["map_route"] = route_data

                    if model is not None:
                        payload = {
                        "Trip_Distance_km": float(route_data["distance_km"]),
                        "Trip_Duration_Minutes": float(route_data["duration_min"]),
                        "Time_of_Day": time_of_day,
                        "Day_of_Week": day_of_week,
                        "Traffic_Conditions": traffic,
                        "Weather": weather,
                    }

                    
                        X_in = build_features(payload)
                        pred_log = float(model.predict(X_in)[0])
                        pred_price_usd = float(np.expm1(pred_log))
                        pred_price_sek = pred_price_usd * USD_TO_SEK

                        st.session_state["map_prediction"] = {
                            "estimated_price": round(pred_price_sek, 2)
                        }
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


if "map_route" in st.session_state and "map_prediction" in st.session_state:
    route = st.session_state["map_route"]
    res = st.session_state["map_prediction"]
    st.success(f"Route Found: {route['distance_km']:.2f} km")

    st.markdown(
        f"""
   <div style='background-color: var(--background-color); padding:5px; border-radius:10px; text-align:center; border:2px solid #3b82f6;'>
         <h1 style='color:#60a5fa; margin:0;'>{res["estimated_price"]:.2f} SEK</h1>
     </div>
    """,
        unsafe_allow_html=True,
    )
    st.caption("Prices are shown in SEK. Use the link below to convert to your local currency:")
    st.markdown("[💱 Convert SEK to any currency](https://www.xe.com/currencyconverter/)", unsafe_allow_html=True)
 
    m = folium.Map(
        location=[
            (route["start_lat"] + route["end_lat"]) / 2,
            (route["start_lon"] + route["end_lon"]) / 2,
        ],
        zoom_start=12,
        zoom_control=False
    )
    folium.PolyLine(route["polyline_latlon"], weight=5, color="#3b82f6").add_to(m)

    with st.container():
        st_folium(m, use_container_width=True, height=400)
