# DiabeaCheck Multi Layer Perceptron dan Integrasi API ML

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

DiabeaCheck Multi Layer Perceptron dan Integrasi API ML adalah sistem machine learning berbasis TensorFlow dan Keras yang dirancang untuk memprediksi risiko diabetes berdasarkan parameter fisiologis. Proyek ini menggunakan Multilayer Perceptron (MLP) yang telah dilatih dan diintegrasikan dengan FastAPI untuk menyediakan endpoint prediksi yang mudah digunakan.

## 📋 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Informasi Model dan Data](#-informasi-model-dan-data)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [API Documentation](#-api-documentation)
- [Cara Kerja](#-cara-kerja)
- [Dependensi](#-dependensi)
- [Development](#-development)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)

## 🚀 Fitur Utama

### 🔬 Model Klasifikasi Diabetes
- Memprediksi kemungkinan seseorang menderita diabetes berdasarkan parameter:
  - **Age** (Usia)
  - **BMI** (Body Mass Index)
  - **Glucose** (Kadar Glukosa)
  - **Insulin** (Kadar Insulin)
  - **BloodPressure** (Tekanan Darah)
- Menggunakan model MLP (Multilayer Perceptron) yang telah dioptimasi

### ⚙️ Preprocessing Data Otomatis
- Penskalaan data input menggunakan `StandardScaler` yang telah dilatih
- Memastikan performa model yang optimal dan konsisten
- Validasi input data secara otomatis

### 🌐 API Endpoint Sederhana
- RESTful API dengan FastAPI
- Endpoint `/predict/` yang mudah diintegrasikan
- Response format JSON yang terstruktur
- Error handling yang comprehensive

## 📊 Informasi Model dan Data

Model ini dilatih menggunakan dataset diabetes dengan teknik machine learning terdepan. Proses pelatihan mencakup:

- **Algoritma**: Multilayer Perceptron (MLP)
- **Framework**: TensorFlow/Keras
- **Preprocessing**: StandardScaler untuk normalisasi data
- **Evaluasi**: Akurasi, Precision, Recall, dan F1-Score

Detail lengkap proses pelatihan dan evaluasi dapat ditemukan di `capstone_diabeacheck.ipynb`.

## 📁 Struktur Proyek

```
DiabeaCheck/
├── app.py                           # Aplikasi utama FastAPI
├── model_artifacts/                 # Artefak model dan preprocessor
│   ├── diabetes_mlp_model.h5       # Model MLP terlatih
│   └── scaler.joblib                # StandardScaler terlatih
├── capstone_diabeacheck.ipynb       # Notebook pelatihan model
├── inference.ipynb                  # Notebook demonstrasi inferensi
├── requirements.txt                 # Dependensi Python
└── README.md                        # Dokumentasi proyek
```

### Deskripsi File

| File | Deskripsi |
|------|-----------|
| `app.py` | Aplikasi utama FastAPI dengan endpoint prediksi |
| `diabetes_mlp_model.h5` | Model MLP yang telah dilatih (TensorFlow/Keras) |
| `scaler.joblib` | Objek StandardScaler untuk penskalaan data |
| `capstone_diabeacheck.ipynb` | Notebook pelatihan dan evaluasi model |
| `inference.ipynb` | Demonstrasi inferensi model secara lokal |
| `requirements.txt` | Daftar dependensi yang diperlukan |

## 🛠️ Instalasi

### Persyaratan Sistem
- **Python**: >= 3.8
- **RAM**: Minimum 4GB (Recommended 8GB)
- **Storage**: Minimum 2GB free space

### Langkah Instalasi

1. **Clone Repository**
   ```bash
   git clone https://github.com/Alpii21/Capstonediabeacheck.git
   cd Capstonediabeacheck/DiabeaCheck
   ```

2. **Buat Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verifikasi Instalasi**
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
   ```

## 🚀 Penggunaan

### Menjalankan Server API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Server akan tersedia di:** `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs`

### Menjalankan Jupyter Notebooks

**Untuk melihat proses pelatihan:**
```bash
jupyter notebook capstone_diabeacheck.ipynb
```

**Untuk demonstrasi inferensi lokal:**
```bash
jupyter notebook inference.ipynb
```

## 📖 API Documentation

### Endpoint: `POST /predict/`

**Deskripsi:** Memprediksi risiko diabetes berdasarkan parameter fisiologis

#### Request Body
```json
{
  "Age": 45,
  "BMI": 28.5,
  "Glucose": 120.0,
  "Insulin": 50.0,
  "BloodPressure": 80
}
```

#### Response Format

**Prediksi Tidak Diabetes:**
```json
{
  "prediction": 0,
  "probability": 0.12345,
  "label": "Tidak Diabetes"
}
```

**Prediksi Diabetes:**
```json
{
  "prediction": 1,
  "probability": 0.87654,
  "label": "Diabetes"
}
```

#### Error Responses

**HTTP 422 - Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "Age"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**HTTP 500 - Server Error:**
```json
{
  "detail": "Model artifacts not found or corrupted"
}
```

### Testing API dengan cURL

```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "BMI": 28.5,
    "Glucose": 120.0,
    "Insulin": 50.0,
    "BloodPressure": 80
  }'
```

## ⚡ Cara Kerja

### Pipeline Prediksi

1. **Input Validation**: Data input divalidasi menggunakan Pydantic models
2. **Data Preprocessing**: Input diskalakan menggunakan `scaler.joblib`
3. **Model Inference**: Data yang sudah diskalakan diproses oleh `diabetes_mlp_model.h5`
4. **Post-processing**: Probabilitas dikonversi menjadi prediksi binary dan label teks
5. **Response**: Hasil dikembalikan dalam format JSON

### Threshold Decision

- **Probability >= 0.5**: Diabetes (Label: "Diabetes")
- **Probability < 0.5**: Tidak Diabetes (Label: "Tidak Diabetes")

## 📦 Dependensi

```txt
fastapi>=0.103.0          # Web framework untuk API
pydantic>=2.0.0           # Data validation
joblib>=1.3.0             # Model serialization
numpy>=1.26.0             # Numerical computing
tensorflow>=2.10.0        # Machine learning framework
pandas>=2.0.0             # Data manipulation
scikit-learn>=1.3.0       # Machine learning utilities
matplotlib>=3.8.0         # Data visualization
seaborn>=0.13.0           # Statistical visualization
imblearn>=0.11.0          # Imbalanced dataset handling
uvicorn>=0.23.0           # ASGI server
```

### Instalasi Dependensi
```bash
pip install -r requirements.txt
```

## 🔬 Development

### Struktur Development

```bash
# Aktivasi virtual environment
source venv/bin/activate  # Linux/macOS
# atau
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt

# Jalankan server dalam mode development
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Testing

**Manual Testing:**
- Gunakan FastAPI interactive docs di `http://localhost:8000/docs`
- Test dengan berbagai kombinasi input parameter

**Unit Testing:**
```bash
# Tambahkan pytest ke requirements untuk testing
pip install pytest pytest-asyncio

# Jalankan tests (jika ada)
pytest tests/
```

### Monitoring Performance

```python
# Contoh monitoring latency
import time

start_time = time.time()
# ... prediksi ...
end_time = time.time()
print(f"Prediction time: {end_time - start_time:.4f} seconds")
```

## 🤝 Kontribusi

Kami menyambut kontribusi dari komunitas! Untuk berkontribusi:

### Guidelines

1. **Fork** repository ini
2. **Create branch** untuk fitur baru (`git checkout -b feature/AmazingFeature`)
3. **Commit** perubahan (`git commit -m 'Add some AmazingFeature'`)
4. **Push** ke branch (`git push origin feature/AmazingFeature`)
5. **Create Pull Request**

### Contribution Areas

- **Model Improvement**: Optimasi akurasi dan performa model
- **API Enhancement**: Penambahan fitur dan endpoint baru
- **Documentation**: Perbaikan dan penambahan dokumentasi
- **Testing**: Penambahan unit tests dan integration tests
- **Performance**: Optimasi kecepatan inference dan memory usage

### Code Standards

- Gunakan **PEP 8** untuk Python code style
- Tambahkan **docstrings** untuk fungsi dan class
- **Type hints** untuk parameter dan return values
- **Error handling** yang comprehensive

## 👥 Tim Pengembang

### 🔎 Machine Learning Team

* **Alfiah** (MC796D5X0076) – *Politeknik Baja Tegal*
* **Elaine Agustina** (MC834D5X1658) – *Universitas Pelita Harapan*
* **Rafly Ashraffi Rachmat** (MC796D5Y0101) – *Politeknik Baja Tegal*

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

```
MIT License

Copyright (c) 2025 DiabeaCheck Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files
