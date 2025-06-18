from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model
from typing import Optional

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path Model dan Scaler
MODEL_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(MODEL_DIR, 'diabetes_mlp_model.h5')  # .h5
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# Fungsi Load Model dan Scaler
def load_artifacts(model_path: str, scaler_path: str):
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        logger.info("✅ Model dan Scaler berhasil dimuat.")
        return model, scaler
    except FileNotFoundError as e:
        logger.error(f"❌ File tidak ditemukan: {e}")
    except Exception as e:
        logger.error(f"❌ Gagal memuat model/scaler: {e}")
    return None, None

# Load model & scaler
model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)

# Inisialisasi FastAPI
app = FastAPI(
    title="DiabeaCheck API",
    version="1.0",
    description="API deteksi risiko diabetes berbasis MLP",
    contact={"name": "Tim DiabeaCheck", "email": "diabeacheck@dbs.academy"}
)

# Schema Input
class PredictionInput(BaseModel):
    Age: int = Field(..., ge=0, le=120)
    BMI: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BloodPressure: int = Field(..., ge=0)

# Schema Output
class PredictionOutput(BaseModel):
    prediction: str
    raw_output: int
    probability_percent: str

# Root
@app.get("/")
async def root():
    return {
        "message": "🎯 DiabeaCheck API - Prediksi Risiko Diabetes",
        "health": "/health",
        "predict": "/predict"
    }

# Endpoint Health
@app.get("/health")
async def health():
    if model and scaler:
        return {"status": "ok", "message": "Model dan Scaler tersedia"}
    return {"status": "error", "message": "Model atau Scaler tidak tersedia"}

# Endpoint Prediksi
@app.post("/predict/", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model atau Scaler tidak tersedia")

    try:
        # Ambil hanya fitur yang dipakai model
        input_data = [
            float(str(data.Age).replace(",", ".")),
            float(str(data.BMI).replace(",", ".")),
            float(str(data.Glucose).replace(",", ".")),
            float(str(data.Insulin).replace(",", ".")),
            float(str(data.BloodPressure).replace(",", "."))
        ]
        logger.info(f"📥 Data diterima: {input_data}")

        input_scaled = scaler.transform([input_data])
        prob = model.predict(input_scaled)[0][0]
        pred = int(prob > 0.5)
        label = "Diabetes" if pred == 1 else "Tidak Diabetes"
        percent = round(prob * 100, 2)

        return {
            "prediction": label,
            "raw_output": pred,
            "probability_percent": f"{percent}%"
        }

    except Exception as e:
        logger.error(f"❌ Prediksi gagal: {e}")
        raise HTTPException(status_code=500, detail="Kesalahan saat memproses prediksi")
