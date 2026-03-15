import sys
import os
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent.parent 

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


import streamlit as st
import numpy as np
import joblib


from src.taxipred.utils.constants import USD_TO_SEK
from src.taxipred.backend.data_processing import build_features


MODEL_PATH = BASE_DIR / "src" / "taxipred" / "backend" / "random_forest_model.joblib"
IMAGE_PATH = BASE_DIR / "src" / "taxipred" / "frontend" / "taxi_image.png"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()


st.set_page_config(page_title="Manual Taxi Predictor", page_icon="🚖")
st.title("🚖 Taxi Price Prediction")

with st.sidebar:
    st.header("Trip Parameters")
    with st.form("predict_form"):
        dist = st.number_input("Distance (km)", min_value=0.1, value=5.0, step=0.1)
        dur = st.number_input("Duration (min)", min_value=1.0, value=15.0, step=1.0)
        time_of_day = st.selectbox(
            "Time of Day", ["Morning", "Afternoon", "Evening", "Night"]
        )
        day_of_week = st.selectbox("Day of Week", ["Weekday", "Weekend"])
        traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
        weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"])
        submitted = st.form_submit_button("Predict Fare")

    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()


if submitted:
    payload = {
        "Trip_Distance_km": float(dist),
        "Trip_Duration_Minutes": float(dur),
        "Time_of_Day": time_of_day,
        "Day_of_Week": day_of_week,
        "Traffic_Conditions": traffic,
        "Weather": weather,
    }

    if model is not None:
        try:
            X_in = build_features(payload)
            pred_log = float(model.predict(X_in)[0])
            pred_price_usd = float(np.expm1(pred_log))
            pred_price_sek = pred_price_usd * USD_TO_SEK
            
            st.session_state["last_prediction"] = {
                "estimated_price": round(pred_price_sek, 2)
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model is not loaded. Check logs.")


col_left, col_right = st.columns([3, 1])

with col_left:
    st.subheader("Predicted price")
    if "last_prediction" in st.session_state:
        res = st.session_state["last_prediction"]
        st.markdown(
            f"""
        <div style='background-color:#1e293b; padding:20px; border-radius:10px; border:2px solid #3b82f6;'>
            <h1 style='color:#60a5fa;'>{res["estimated_price"]:.2f} SEK</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.info('Adjust parameters in the sidebar and click "Predict Fare" to see the result.')

    st.write("")
    
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, use_container_width=False)
    
    st.markdown(
        """
    <p style='text-align: center; font-size:28px;'>
    Developed by MLOps student <a href='https://www.linkedin.com/in/lilit-ajoyan-1565b4183/' target='_blank' style='color: #60a5fa; text-decoration: underline;'>
      Lilit Ajoyan 
      </a>
      <br>
     Find the repo on 
    <a href='https://github.com/LAjoyan/taxi_prediction_fullstack_lilit' target='_blank' style='color: #60a5fa; text-decoration: underline;'>
            GitHub
        </a>
    </p>
    """,
        unsafe_allow_html=True,
    )

with col_right:
    st.subheader("System Status")
    if model is not None:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Offline")
    
    st.divider()
    st.caption("Running on Streamlit Cloud (Direct Mode)")