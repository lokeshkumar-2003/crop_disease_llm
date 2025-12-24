import requests
from config.apiConfig import LOCATION_API

def get_coordinates(location_name: str):
    response = requests.get(f"{LOCATION_API}&q={location_name}")
    data = response.json()

    if not data:
        raise Exception("Location not found")

    return data[0]["lat"], data[0]["lon"], data[0]["name"]
