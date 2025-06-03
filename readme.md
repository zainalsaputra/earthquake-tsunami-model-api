
# 🌊 Earthquake & Tsunami Prediction API (Indonesia)

A lightweight and scalable **FastAPI-based** server that predicts the **earthquake disaster status and tsunami potential** in Indonesia using **TensorFlow Lite (TFLite)**. This application is part of a responsive web platform aiming to close the information gap about earthquake and tsunami risks by combining real-time geospatial data, weather forecasts, and machine learning models.

---

## 📦 Features

- 🌍 Predicts disaster risk (`status`) based on geolocation (latitude, longitude).
- 🌪️ Retrieves live weather data via [Open-Meteo API](https://open-meteo.com).
- 🌐 Uses geolocation data via [Nominatim (OpenStreetMap)](https://nominatim.org).
- 🤖 Model inference with lightweight `.tflite` deep learning model.
- 🔧 Input features are automatically preprocessed with a pre-trained `preprocessor.pkl`.
- ✅ Returns disaster status, magnitude, depth, and confidence score.

---

## 📁 Project Structure

```
.
├── app.py
├── models/
│   ├── model_dnn.tflite
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
```

---

## 🚀 How to Run

### 1. Clone this repository

```bash
git clone https://github.com/zainalsaputra/mitigasi-kita-app-model.git
cd mitigasi-kita-app-model
```

### 2. Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### 3. Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 3000 --reload
```

> Server will run at: `http://localhost:3000`

---

## 🔍 Example Usage

### Request

`POST /predict`

```json
{
  "latitude": -6.200000,
  "longitude": 106.816666
}
```

### Response

```json
{
  "status": "success",
  "data": {
    "location": "Jawa Barat",
    "city": "Jakarta",
    "agency": "BMKG",
    "mag_type": "M",
    "magnitude": 4.427,
    "depth": 28.0,
    "azimuth_gap": 136.0,
    "phasecount": 65,
    "potensi_tsunami": "Rendah",
    "latitude": -6.2,
    "longitude": 106.816666,
    "temperature_2m_min": 25.3,
    "temperature_2m_max": 29.5,
    "windspeed_10m_max": 21.7,
    "precipitation_sum": 17.9,
    "status": "Rendah",
    "confidence_score": 0.872
  }
}
```

---

## 📚 Dependencies

```
fastapi==0.103.0
geopy==2.4.1
joblib==1.5.1
numpy==2.1.3
pandas==2.2.3
requests==2.32.3
tensorflow==2.19.0
scikit-learn==1.6.1
```

> ✅ You may replace `tensorflow` with `tflite-runtime` for lightweight environments:

```bash
pip uninstall tensorflow
pip install tflite-runtime
```

And update the import in code:

```python
import tflite_runtime.interpreter as tflite
```

---

## 🧠 Model Information

- Trained using a deep neural network (DNN) with TensorFlow.
- Converted to `.tflite` format for efficient inference.
- Classification model for disaster severity: `Rendah`, `Sedang`, `Tinggi`.

---

## 📄 License

MIT License - Use freely with attribution.

---

## ✨ Credits

Developed as part of an initiative to support **early warning systems** and **disaster mitigation** technology in Indonesia.
