import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

API_URL = 'http://127.0.0.1:8000/api/taxi/v1'

st.set_page_config(page_title='Address Predictor', page_icon='📍')
st.title('📍 Address-to-Address Prediction')

with st.sidebar:
    st.header('Conditions')
    time_of_day = st.selectbox(
        'Time of Day', ['Morning', 'Afternoon', 'Evening', 'Night']
    )
    day_of_week = st.selectbox('Day of Week', ['Weekday', 'Weekend'])
    traffic = st.selectbox('Traffic', ['Low', 'Medium', 'High'])
    weather = st.selectbox('Weather', ['Clear', 'Rain', 'Snow'])

col1, col2 = st.columns(2)
with col1:
    from_address = st.text_input('From (Address A)')
with col2:
    to_address = st.text_input('To (Address B)')

if st.button('Calculate Route & Predict Fare'):
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
    <div style='background-color:#1e293b; padding:25px; border-radius:15px; text-align:center; border:2px solid #3b82f6;'>
         <h1 style='color:#60a5fa; margin:0;'>{res['estimated_price']:.2f} SEK</h1>
     </div>
            ''',
        unsafe_allow_html=True,
    )
