import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# List of cities with coordinates
cities = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
    "New York", "London", "Tokyo", "Sydney", "Cairo",
    "Moscow", "Rio de Janeiro", "Cape Town", "Dubai", "Singapore"
]

# Open-Meteo free API (no API key required)
base_url = "https://api.open-meteo.com/v1/forecast"

weather_data = []

for city in cities:
    # Approximate coordinates (you could use geocoding API for precise)
    coords = {
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Bangalore": (12.9716, 77.5946),
        "Chennai": (13.0827, 80.2707),
        "Kolkata": (22.5726, 88.3639),
        "New York": (40.7128, -74.0060),
        "London": (51.5074, -0.1278),
        "Tokyo": (35.6895, 139.6917),
        "Sydney": (-33.8688, 151.2093),
        "Cairo": (30.0444, 31.2357),
        "Moscow": (55.7558, 37.6173),
        "Rio de Janeiro": (-22.9068, -43.1729),
        "Cape Town": (-33.9249, 18.4241),
        "Dubai": (25.2048, 55.2708),
        "Singapore": (1.3521, 103.8198)
    }
    
    lat, lon = coords[city]
    
    # API call for current weather
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m"
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if "current_weather" in data:
        weather = data["current_weather"]
        weather_data.append({
            "city": city,
            "temperature": weather["temperature"],
            "windspeed": weather["windspeed"],
            "latitude": lat,
            "longitude": lon
        })

df = pd.DataFrame(weather_data)
print("Raw data (no labels):")
print(df)


