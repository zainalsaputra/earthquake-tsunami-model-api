from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import requests
import tensorflow as tf
import pandas as pd
from geopy.geocoders import Nominatim
from typing import Optional

# === INIT APP ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === PATHS ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
TFLITE_PATH = os.path.join(MODEL_DIR, 'model_dnn.tflite')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# === LOAD MODEL ===
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === LOAD PREPROCESSOR ===
preprocessor = joblib.load(PREPROCESSOR_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# === GEO CODER ===
geocoder = Nominatim(user_agent="mitigasi_kita")

# === MODELS ===
class Coordinate(BaseModel):
    latitude: float
    longitude: float

# === UTILITY FUNCTIONS ===
def get_weather_data(latitude, longitude):
    try:
        url = f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max&timezone=auto'
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        daily = response.json()['daily']
        return {
            'temperature_2m_max': daily['temperature_2m_max'][0],
            'temperature_2m_min': daily['temperature_2m_min'][0],
            'precipitation_sum': daily['precipitation_sum'][0],
            'windspeed_10m_max': daily['windspeed_10m_max'][0],
            'weathercode': 800  
        }
    except:
        return {
            'temperature_2m_max': 29.5,
            'temperature_2m_min': 25.3,
            'precipitation_sum': 17.9,
            'windspeed_10m_max': 21.7,
            'weathercode': 800
        }

def get_location_data(latitude, longitude):
    try:
        location = geocoder.reverse((latitude, longitude), language="id")
        address = location.raw.get("address", {}) if location else {}
        city = address.get("city") or address.get("town") or address.get("village") or "Jakarta"
        location_name = address.get("state", "Jawa")
        return location_name, city
    except:
        return "Jawa", "Jakarta"

def determine_tsunami_potential(magnitude, depth):
    if magnitude >= 1.6 and depth < 70:  # Dangkal dan magnitudo tertinggi
        return "Tinggi", "Bahaya"
    elif magnitude >= 1.1 and depth < 300:  # Magnitudo sedang, kedalaman menengah
        return "Sedang", "Waspada"
    else:  # Magnitudo rendah atau kedalaman lebih dalam
        return "Rendah", "Aman"

def prepare_input_data(latitude, longitude, predicted_class="Aman"):
    weather = get_weather_data(latitude, longitude)
    location, city = get_location_data(latitude, longitude)

    # Generate magnitude based on predicted_class
    if predicted_class == "Aman":
        magnitude = np.random.uniform(0.0, 1.0)
    elif predicted_class == "Waspada":
        magnitude = np.random.uniform(1.1, 1.5)
    else:  # Bahaya
        magnitude = np.random.uniform(1.6, 2.5)

    # Generate depth based on predicted_class
    if predicted_class == "Aman":
        depth = np.random.uniform(301.0, 700.0)  # Menengah, risiko rendah
    elif predicted_class == "Waspada":
        depth = np.random.uniform(71.0, 300.0)  # Dangkal-menengah, risiko sedang
    else:  # Bahaya
        depth = np.random.uniform(0.0, 70.0)  # Dangkal, risiko lebih tinggi

    quake = {
        'magnitude': magnitude,
        'mag_type': 'M',
        'depth': depth,
        'phasecount': 65,
        'azimuth_gap': 136.0,
        'potensi_tsunami': 'Rendah'
    }

    other = {
        'location': location,
        'agency': 'BMKG',
        'city': city,
        'potensi_gempa': 'Rendah'
    }

    full_input = {
        'latitude': latitude,
        'longitude': longitude,
        **weather,
        **quake,
        **other
    }

    return full_input  

# === ROUTE ===
@app.post("/predict")
async def predict(coord: Coordinate):
    try:
        latitude = coord.latitude
        longitude = coord.longitude

        input_dict = prepare_input_data(latitude, longitude, "Aman")
        df_input = pd.DataFrame([input_dict])
        df_input = df_input[preprocessor.feature_names_in_]
        X = preprocessor.transform(df_input).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], X)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class = label_encoder.inverse_transform([class_idx])[0]

        input_dict = prepare_input_data(latitude, longitude, predicted_class)
        df_input = pd.DataFrame([input_dict])
        df_input = df_input[preprocessor.feature_names_in_]
        X = preprocessor.transform(df_input).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], X)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class = label_encoder.inverse_transform([class_idx])[0]
        confidence_score = float(prediction[0][class_idx])

        magnitude = input_dict['magnitude']
        depth = input_dict['depth']
        potensi_tsunami, _ = determine_tsunami_potential(magnitude, depth)

        return {
            'status': 'success',
            'data': {
                'location': input_dict['location'],
                'city': input_dict['city'],
                'agency': input_dict['agency'],
                'mag_type': input_dict['mag_type'],
                'magnitude': float(magnitude),
                'depth': float(depth),
                'azimuth_gap': float(input_dict['azimuth_gap']),
                'phasecount': int(input_dict['phasecount']),
                'potensi_tsunami': potensi_tsunami,
                'latitude': latitude,
                'longitude': longitude,
                'temperature_2m_min': float(input_dict['temperature_2m_min']),
                'temperature_2m_max': float(input_dict['temperature_2m_max']),
                'windspeed_10m_max': float(input_dict['windspeed_10m_max']),
                'precipitation_sum': float(input_dict['precipitation_sum']),
                'status': predicted_class,
                #'confidence_score': confidence_score
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def home(request: Request):
    base_url = str(request.base_url)
    return {
        "status": "success",
        "message": f"Welcome to Earthquake & Tsunami Prediction API, show documentation at {base_url}docs"
    }

# uvicorn app:app --host 0.0.0.0 --port 5000 --reload
