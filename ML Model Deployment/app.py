from fastapi import FastAPI, HTTPException
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

# ─── Fungsi untuk Load Model dan Scaler ────────────────────────────────
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

# ─── Load Model dan Scaler ─────────────────────────────────────────────
model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)

# ─── Inisialisasi FastAPI ─────────────────────────────────────────────
app = FastAPI(
    title="DiabeaCheck API",
    version="1.0",
    description="🎯 API untuk deteksi dini risiko diabetes menggunakan model MLP.",
    contact={"name": "Tim DiabeaCheck", "email": "diabeacheck@dbs.academy"}
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
        "message": "🎯 DiabeaCheck API - Prediksi Risiko Diabetes",
        "health": "/health",
        "predict": "/predict"
    }

# ─── Endpoint Kesehatan Sistem ─────────────────────────────────────────
@app.get("/health", status_code=200)
async def health_check():
    if model is not None and scaler is not None:
        return {"status": "ok", "message": "Model dan Scaler tersedia"}
    return {"status": "error", "message": "Model atau Scaler tidak tersedia"}

# ─── Endpoint Prediksi ─────────────────────────────────────────────────
@app.post("/predict/", response_model=PredictionOutput, status_code=200)
async def predict_diabetes(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model atau Scaler tidak tersedia")

    try:
        # Ambil data input
        input_data = [
            float(str(data.Age).replace(",", ".")),
            float(str(data.BMI).replace(",", ".")),
            float(str(data.Glucose).replace(",", ".")),
            float(str(data.Insulin).replace(",", ".")),
            float(str(data.BloodPressure).replace(",", "."))
        ]

        logger.info(f"📥 Data diterima: {input_data}")
        input_array = np.array([input_data])

        # Scaling
        try:
            input_scaled = scaler.transform(input_array)
        except Exception as scale_err:
            logger.error(f"❌ Error saat scaling: {scale_err}")
            raise HTTPException(status_code=500, detail="Kesalahan saat scaling data")

        # Prediksi
        prob = model.predict(input_scaled)
        prob_val = float(prob[0][0]) if prob.shape[-1] == 1 else float(prob[0])
        pred = int(prob_val > 0.5)
        label = "Diabetes" if pred == 1 else "Tidak Diabetes"
        percent = round(prob_val * 100, 2)

        logger.info(f"📤 Prediksi: {label} | Probabilitas: {percent}%")

        return {
            "prediction": label,
            "raw_output": pred,
            "probability_percent": f"{percent}%"
        }

    except Exception as e:
        logger.error(f"❌ Gagal melakukan prediksi: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat memproses prediksi")
