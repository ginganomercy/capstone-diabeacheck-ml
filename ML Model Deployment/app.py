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

# ─── Load Artifacts ────────────────────────────────────────────────────
model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)

# ─── Inisialisasi FastAPI ─────────────────────────────────────────────
app = FastAPI(
    title="DiabeaCheck API",
    version="1.0",
    description="🎯 API untuk mendeteksi dini risiko diabetes menggunakan model MLP.",
    contact={
        "name": "Tim DiabeaCheck",
        "email": "diabeacheck@dbs.academy"
    }
)

# ─── Schema Input ──────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    Age: int = Field(..., ge=0, le=120, example=0)
    BMI: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BloodPressure: int = Field(..., ge=0)

    # Opsi tambahan yang tidak wajib
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
    if model and scaler:
        return {"status": "ok", "message": "Model dan Scaler tersedia"}
    else:
        return {"status": "error", "message": "Model atau Scaler tidak dimuat dengan benar"}

# ─── Endpoint Prediksi ─────────────────────────────────────────────────
@app.post("/predict/", response_model=PredictionOutput, status_code=200)
async def predict_diabetes(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model atau scaler gagal dimuat saat startup.")

    try:
        # Konversi ke dict untuk manipulasi nilai
        data_dict = data.dict()

        # Fungsi bantu untuk parsing float dengan koma
        def parse_value(val):
            if isinstance(val, str):
                val = val.replace(",", ".")
            return float(val)

        # Parsing nilai input (wajib)
        age = data_dict['Age']
        bmi = parse_value(data_dict['BMI'])
        glucose = parse_value(data_dict['Glucose'])
        insulin = parse_value(data_dict['Insulin'])
        blood_pressure = data_dict['BloodPressure']

        # Siapkan input array
        input_array = np.array([[age, bmi, glucose, insulin, blood_pressure]])

        logger.info(f"📥 Input diterima: {data_dict}")

        # Transformasi dan prediksi
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
        logger.error(f"❌ Gagal melakukan prediksi: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat melakukan prediksi.")
