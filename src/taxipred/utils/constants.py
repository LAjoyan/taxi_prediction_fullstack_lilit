from pathlib import Path
import os
import requests

ORS_API_KEY = os.getenv("ORS_API_KEY")

USD_TO_SEK = 10.5

def get_route_data(from_addr, to_addr):

    if not ORS_API_KEY:
        raise ValueError("ORS_API_KEY not found in environment variables.")


    def geocode(addr):
        url = f"https://api.openrouteservice.org/geocode/search?api_key={ORS_API_KEY}&text={addr}"
        resp = requests.get(url).json()
        if not resp.get('features'):
            raise ValueError(f"Could not find address: {addr}")
        return resp['features'][0]['geometry']['coordinates'] 
    

    coords_from = geocode(from_addr)
    coords_to = geocode(to_addr)


    route_url = f"https://api.openrouteservice.org/v2/directions/driving-car?api_key={ORS_API_KEY}&start={coords_from[0]},{coords_from[1]}&end={coords_to[0]},{coords_to[1]}"
    route_resp = requests.get(route_url).json()

    if 'features' not in route_resp:
        raise ValueError("Could not calculate route.")

    properties = route_resp['features'][0]['properties']['summary']
    geometry = route_resp['features'][0]['geometry']['coordinates']

    return {
        "distance_km": properties['distance'] / 1000,
        "duration_min": properties['duration'] / 60,
        "start_lat": coords_from[1],
        "start_lon": coords_from[0],
        "end_lat": coords_to[1],
        "end_lon": coords_to[0],
        "polyline_latlon": [[c[1], c[0]] for c in geometry] # Flip for Folium [lat, lon]
    }

DATA_PATH = Path(__file__).parents[1] / "data"
