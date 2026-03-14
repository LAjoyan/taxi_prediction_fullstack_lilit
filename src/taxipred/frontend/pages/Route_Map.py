import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

API_URL = 'http://127.0.0.1:8000/api/taxi/v1'

st.set_page_config(page_title='Address Predictor', page_icon='📍')
# CSS to remove extra whitespace and fix the title position
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
        }
        h1 {
            /* Changed from negative to positive to push it down */
            margin-top: 2rem !important; 
            margin-bottom: 1.5rem !important;
            text-align: left; /* Adjust to 'center' if you prefer */
        }
    </style>
""", unsafe_allow_html=True)


st.title('📍 Address-to-Address Prediction')

with st.sidebar:
    st.header('Conditions')
    time_of_day = st.selectbox(
        'Time of Day', ['Morning', 'Afternoon', 'Evening', 'Night']
    )
    day_of_week = st.selectbox('Day of Week', ['Weekday', 'Weekend'])
    traffic = st.selectbox('Traffic', ['Low', 'Medium', 'High'])
    weather = st.selectbox('Weather', ['Clear', 'Rain', 'Snow'])


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
    # label_visibility="collapsed" hides the "From" text above the box to save space
    from_address = st.text_input("From Address", placeholder="Enter starting point...", label_visibility="collapsed")

with col_addr2:
    to_address = st.text_input("To Address", placeholder="Enter destination...", label_visibility="collapsed")

with col_btn:
    # This puts the button on the same line as the inputs
    predict_clicked = st.button('Predict Fare', use_container_width=True)

# Now, use 'predict_clicked' instead of 'if st.button(...)'
if predict_clicked:
    if not from_address or not to_address:
        st.warning('Please enter both addresses.')

    else:
        try:
            with st.spinner('Calculating ...'):
                r = requests.post(
                    f'{API_URL}/route',
                    json={'from_address': from_address, 'to_address': to_address},
                )
                r.raise_for_status()
                route_data = r.json()
                
                if route_data['distance_km'] > 100:
                    st.error('🚨 Distance too far! This model is only for city trips under 100km.')
                    st.warning(f'Calculated distance: {route_data['distance_km']:.2f} km is unrealistic for a taxi.')
                    if 'map_route' in st.session_state: 
                        del st.session_state['map_route']
                    if 'map_prediction' in st.session_state: 
                        del st.session_state['map_prediction']
                else:
                    st.session_state['map_route'] = route_data

                    payload = {
                        'Trip_Distance_km': float(route_data['distance_km']),
                        'Trip_Duration_Minutes': float(route_data['duration_min']),
                        'Time_of_Day': time_of_day,
                        'Day_of_Week': day_of_week,
                        'Traffic_Conditions': traffic,
                        'Weather': weather,
                    }

                    p = requests.post(f'{API_URL}/predict', json=payload)
                    p.raise_for_status()
                    st.session_state['map_prediction'] = p.json()

        except Exception as e:
            st.error(f'Error: {e}')


if 'map_route' in st.session_state and 'map_prediction' in st.session_state:
    route = st.session_state['map_route']
    res = st.session_state['map_prediction']
    st.success(f'Route Found: {route['distance_km']:.2f} km')

    m = folium.Map(
        location=[
            (route['start_lat'] + route['end_lat']) / 2,
            (route['start_lon'] + route['end_lon']) / 2,
        ],
        zoom_start=12,
    )
    folium.PolyLine(route['polyline_latlon'], weight=5).add_to(m)
    st_folium(m, width=700, height=400)

    st.markdown(
        f'''
    <div style='background-color:#1e293b; padding:5px; border-radius:10px; text-align:center; border:1px solid #3b82f6;'>
         <h1 style='color:#60a5fa; margin:0;'>{res['estimated_price']:.2f} SEK</h1>
     </div>
            ''',
        unsafe_allow_html=True,
    )


