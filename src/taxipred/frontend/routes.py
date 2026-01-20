import os
import requests

ORS_API_KEY = os.getenv("ORS_API_KEY")
ORS_BASE_URL = "https://api.openrouteservice.org"


def geocode_address(query: str):
    if not ORS_API_KEY:
        raise RuntimeError("Missing ORS_API_KEY environment variable.")

    url = f"{ORS_BASE_URL}/geocode/search"
    params = {"text": query, "size": 1}
    headers = {"Authorization": ORS_API_KEY}

    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()

    features = data.get("features", [])
    if not features:
        raise ValueError(f"No result for: {query}")

    lon, lat = features[0]["geometry"]["coordinates"]
    return lat, lon


def get_route(start_latlon, end_latlon):
    """
    Gets a driving route between 2 points.
    Returns: geometry(list of [lat, lon]), distance_km, duration_min
    """
    if not ORS_API_KEY:
        raise RuntimeError("Missing ORS_API_KEY environment variable.")

    url = f"{ORS_BASE_URL}/v2/directions/driving-car"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}

    start_lat, start_lon = start_latlon
    end_lat, end_lon = end_latlon

    body = {
        "coordinates": [
            [start_lon, start_lat],
            [end_lon, end_lat],
        ]
    }

    r = requests.post(url, json=body, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    route = data["features"][0]
    summary = route["properties"]["summary"]
    distance_km = summary["distance"] / 1000
    duration_min = summary["duration"] / 60

    coords_lonlat = route["geometry"]["coordinates"]
    geometry_latlon = [[lat, lon] for lon, lat in coords_lonlat]

    return geometry_latlon, distance_km, duration_min
