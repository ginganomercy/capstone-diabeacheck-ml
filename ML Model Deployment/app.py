from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model

# ─── Konfigurasi Logging ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Path Model & Scaler ───────────────────────────────────────────────
MODEL_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(MODEL_DIR, 'diabetes_mlp_model.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# ─── Load Model dan Scaler ─────────────────────────────────────────────
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("✅ Model dan Scaler berhasil dimuat.")
except FileNotFoundError as e:
    logger.error(f"❌ Gagal memuat model atau scaler: {e}")
    model = None
    scaler = None

# ─── Inisialisasi FastAPI ─────────────────────────────────────────────
app = FastAPI(title="DiabeaCheck API", version="1.0")

# ─── Schema Input ──────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    Age: int = Field(..., ge=0, le=120, example=25)
    BMI: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BloodPressure: int = Field(..., ge=0)

# ─── Endpoint Prediksi ─────────────────────────────────────────────────
@app.post("/predict/")
async def predict_diabetes(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model atau scaler gagal dimuat saat startup.")

    input_array = np.array([[data.Age, data.BMI, data.Glucose, data.Insulin, data.BloodPressure]])
    logger.info(f"📥 Data diterima untuk prediksi: {input_array.tolist()}")

    input_scaled = scaler.transform(input_array)
    probability = float(model.predict(input_scaled)[0][0])
    prediction = int(probability > 0.5)
    label = "Diabetes" if prediction == 1 else "Tidak Diabetes"

    logger.info(f"📤 Prediksi: {label} (Probabilitas: {probability:.4f})")

    return {
        "prediction": label,
        "raw_output": prediction,
        "probability": round(probability, 4)
    }

# ─── Endpoint Root ─────────────────────────────────────────────────────
@app.get("/")
async def read_root():
    return {
        "message": "🎯 DiabeaCheck API - Deteksi Dini Risiko Diabetes menggunakan MLP Model",
        "health": "/health",
        "predict": "/predict"
    }

# ─── Endpoint Health Check ─────────────────────────────────────────────
@app.get("/health")
async def health_check():
    if model and scaler:
        return {"status": "ok", "message": "Model dan Scaler tersedia"}
    else:
        return {"status": "error", "message": "Model atau Scaler tidak dimuat dengan benar"}
