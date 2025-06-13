from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model
from typing import Optional

# ─── Konfigurasi Logging ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Path Model & Scaler ───────────────────────────────────────────────
MODEL_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(MODEL_DIR, 'diabetes_mlp_model.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

model = None
scaler = None

# ─── Fungsi untuk Load Model dan Scaler ────────────────────────────────
def load_artifacts():
    global model, scaler
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Model dan Scaler berhasil dimuat.")
    except FileNotFoundError as e:
        logger.error(f"❌ File tidak ditemukan: {e}")
    except Exception as e:
        logger.error(f"❌ Gagal memuat model/scaler: {e}")

# ─── Panggil Fungsi Saat Startup ───────────────────────────────────────
load_artifacts()

# ─── Inisialisasi FastAPI ─────────────────────────────────────────────
app = FastAPI(
    title="DiabeaCheck API",
    version="1.0",
    description="🎯 API untuk deteksi dini risiko diabetes menggunakan model MLP.",
    contact={
        "name": "Tim DiabeaCheck",
        "email": "diabeacheck@dbs.academy"
    }
)

# ─── Middleware CORS ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schema Input ──────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    Age: int = Field(..., ge=0, le=120)
    BMI: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BloodPressure: int = Field(..., ge=0)
    Pregnancies: Optional[int] = Field(None, ge=0)
    SkinThickness: Optional[float] = Field(None, ge=0)
    DiabetesPedigreeFunction: Optional[float] = Field(None, ge=0)

# ─── Schema Output ─────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    prediction: str
    raw_output: int
    probability_percent: str

# ─── Endpoint Root ─────────────────────────────────────────────────────
@app.get("/", status_code=200)
async def read_root():
    return {
        "message": "🎯 DiabeaCheck API - Deteksi Dini Risiko Diabetes menggunakan MLP Model",
        "health": "/health",
        "predict": "/predict"
    }

# ─── Endpoint Health Check ─────────────────────────────────────────────
@app.get("/health", status_code=200)
async def health_check():
    if model is not None and scaler is not None:
        logger.info("✅ Health check sukses")
        return {"status": "ok", "message": "Model dan Scaler tersedia"}
    else:
        logger.error("❌ Model/Scaler tidak tersedia saat health check")
        raise HTTPException(status_code=500, detail="Model atau Scaler tidak tersedia.")

# ─── Endpoint Prediksi ─────────────────────────────────────────────────
@app.post("/predict/", response_model=PredictionOutput, status_code=200)
async def predict_diabetes(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model atau Scaler belum dimuat.")

    try:
        input_array = np.array([[data.Age, data.BMI, data.Glucose, data.Insulin, data.BloodPressure]])
        logger.info(f"📥 Input diterima: {data.dict()}")

        input_scaled = scaler.transform(input_array)
        probability = float(model.predict(input_scaled)[0][0])
        prediction = int(probability > 0.5)
        label = "Diabetes" if prediction == 1 else "Tidak Diabetes"
        percent = round(probability * 100, 2)

        logger.info(f"📤 Prediksi: {label} | Probabilitas: {percent}%")

        return {
            "prediction": label,
            "raw_output": prediction,
            "probability_percent": f"{percent}%"
        }

    except Exception as e:
        logger.error(f"❌ Error saat prediksi: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat melakukan prediksi.")
