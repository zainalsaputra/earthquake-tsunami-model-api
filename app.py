from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
import requests
from geopy.geocoders import Nominatim
from flask_cors import CORS
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
CORS(app)

# === PATHS ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
TFLITE_PATH = os.path.join(MODEL_DIR, 'model_dnn.tflite')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# === LOAD MODEL ===
interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === LOAD PREPROCESSOR ===
preprocessor = joblib.load(PREPROCESSOR_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# === GEO CODER ===
geocoder = Nominatim(user_agent="mitigasi_kita")

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
            'weathercode': 800  # fallback
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
        city = address.get("city") or address.get("town") or address.get("village") or "Ambon"
        location_name = address.get("state", "Banda Sea")
        return location_name, city
    except:
        return "Banda Sea", "Ambon"

def determine_tsunami_potential(magnitude, depth):
    if magnitude > 7 and depth < 30:
        return "Tinggi", "Bahaya"
    elif magnitude > 6:
        return "Sedang", "Waspada"
    else:
        return "Rendah", "Aman"

def prepare_input_data(latitude, longitude):
    weather = get_weather_data(latitude, longitude)
    location, city = get_location_data(latitude, longitude)

    # Data gempa statis simulasi
    quake = {
        'magnitude': 4.427,
        'mag_type': 'M',
        'depth': 28.0,
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

    return pd.DataFrame([full_input], columns=preprocessor.feature_names_in_)

# === ROUTES ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))

        input_df = prepare_input_data(latitude, longitude)
        X = preprocessor.transform(input_df).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], X)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class = label_encoder.inverse_transform([class_idx])[0]
        confidence_score = float(prediction[0][class_idx])

        magnitude = input_df['magnitude'].iloc[0]
        depth = input_df['depth'].iloc[0]
        potensi_tsunami, _ = determine_tsunami_potential(magnitude, depth)

        return jsonify({
            'status': 'success',
            'data': {
                'location': input_df['location'].iloc[0],
                'city': input_df['city'].iloc[0],
                'agency': 'BMKG',
                'mag_type': 'M',
                'magnitude': float(magnitude),
                'depth': float(depth),
                'azimuth_gap': float(input_df['azimuth_gap'].iloc[0]),
                'phasecount': int(input_df['phasecount'].iloc[0]),
                'potensi_tsunami': potensi_tsunami,
                'latitude': latitude,
                'longitude': longitude,
                'temperature_2m_min': float(input_df['temperature_2m_min'].iloc[0]),
                'temperature_2m_max': float(input_df['temperature_2m_max'].iloc[0]),
                'windspeed_10m_max': float(input_df['windspeed_10m_max'].iloc[0]),
                'precipitation_sum': float(input_df['precipitation_sum'].iloc[0]),
                'status': predicted_class,
                'confidence_score': confidence_score
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# === RUN SERVER ===
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
