import requests
import os
from dotenv import load_dotenv

load_dotenv()

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(lat: float, lon: float):
    if not API_KEY:
        raise Exception("OPENWEATHER_API_KEY not set")

    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(WEATHER_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Weather API HTTP error: {response.status_code}")

    data = response.json()

    rain_mm = data.get("rain", {}).get("1h", 0)

    # REALISTIC rainfall classification
    if rain_mm == 0:
        rainfall = "no rain"
    elif rain_mm <= 2.5:
        rainfall = "light rain"
    elif rain_mm <= 7.5:
        rainfall = "moderate rain"
    else:
        rainfall = "heavy rain"

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rainfall": rainfall,
        "rain_mm": rain_mm,
        "weather_desc": data["weather"][0]["description"]
    }
