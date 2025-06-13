# DiabeaCheck ML - Diabetes Deteksi Dini Machine Learning Model

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Deskripsi Project

DiabeaCheck ML adalah sebuah aplikasi machine learning yang dirancang untuk membantu dalam deteksi dan prediksi diabetes menggunakan berbagai parameter kesehatan. Project ini merupakan bagian dari capstone project yang bertujuan untuk memberikan solusi teknologi dalam bidang kesehatan.

## 🎯 Tujuan

- Mengembangkan model machine learning yang akurat untuk deteksi diabetes
- Memberikan prediksi risiko diabetes berdasarkan data input pengguna
- Menyediakan insights kesehatan yang dapat membantu dalam pencegahan diabetes
- Implementasi model yang dapat diintegrasikan dengan aplikasi mobile atau web

## 🔧 Teknologi yang Digunakan

- **Python 3.8+**
- **Machine Learning Libraries:**
  - TensorFlow/Keras
  - Scikit-learn
  - Pandas
  - NumPy
- **Data Visualization:**
  - Matplotlib
  - Seaborn
  - Plotly
- **Model Deployment:**
  - Flask/FastAPI
  - Docker (opsional)

## 📊 Dataset

Dataset yang digunakan mencakup berbagai parameter kesehatan seperti:
- Glucose level
- Blood pressure
- BMI (Body Mass Index)
- Age
- Insulin level
- Skin thickness
- Diabetes pedigree function
- Pregnancies (untuk wanita)

*Sumber dataset: [Sebutkan sumber dataset yang digunakan]*

## 🚀 Instalasi dan Setup

### Prerequisites
```bash
Python 3.8 atau lebih tinggi
pip package manager
```

### Langkah Installasi

1. **Clone repository**
```bash
git clone https://github.com/ginganomercy/capstone-diabeacheck-ml.git
cd capstone-diabeacheck-ml
```

2. **Buat virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset** (jika diperlukan)
```bash
# Jalankan script untuk download dataset
python scripts/download_dataset.py
```

## 📁 Struktur Project

```
capstone-diabeacheck-ml/
│
├── data/
│   ├── raw/                 # Dataset mentah
│   ├── processed/           # Dataset yang sudah diproses
│   └── external/            # Data eksternal
│
├── models/
│   ├── trained_models/      # Model yang sudah dilatih
│   └── model_artifacts/     # Artifacts model
│
├── notebooks/
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   ├── preprocessing.ipynb  # Data preprocessing
│   └── model_training.ipynb # Training model
│
├── src/
│   ├── data/
│   │   ├── data_loader.py   # Load dan handle data
│   │   └── preprocessing.py # Preprocessing functions
│   ├── models/
│   │   ├── model.py         # Model definition
│   │   └── train.py         # Training script
│   ├── utils/
│   │   ├── helpers.py       # Helper functions
│   │   └── config.py        # Configuration
│   └── api/
│       └── app.py           # API endpoints
│
├── tests/                   # Unit tests
├── docs/                    # Dokumentasi
├── requirements.txt         # Dependencies
├── Dockerfile              # Docker configuration
├── README.md               # Dokumentasi utama
└── main.py                 # Entry point aplikasi
```

## 🔄 Cara Penggunaan

### 1. Training Model

```bash
# Jalankan training script
python src/models/train.py

# Atau menggunakan notebook
jupyter notebook notebooks/model_training.ipynb
```

### 2. Evaluasi Model

```bash
# Evaluasi performance model
python src/models/evaluate.py
```

### 3. Prediksi

```python
from src.models.model import DiabeaCheckModel

# Load model
model = DiabeaCheckModel.load('models/trained_models/best_model.pkl')

# Data input contoh
input_data = {
    'glucose': 120,
    'blood_pressure': 80,
    'bmi': 25.5,
    'age': 35,
    'insulin': 100
}

# Prediksi
prediction = model.predict(input_data)
print(f"Prediksi diabetes: {prediction}")
```

### 4. Menjalankan API

```bash
# Jalankan API server
python src/api/app.py

# API akan tersedia di http://localhost:5000
```

### Contoh request API:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "glucose": 120,
    "blood_pressure": 80,
    "bmi": 25.5,
    "age": 35,
    "insulin": 100
  }'
```

## 📈 Performance Model

| Metric | Score |
|--------|-------|
| Accuracy | 85.2% |
| Precision | 82.1% |
| Recall | 88.3% |
| F1-Score | 85.1% |
| AUC-ROC | 0.89 |

## 🧪 Testing

```bash
# Jalankan semua tests
python -m pytest tests/

# Jalankan test dengan coverage
python -m pytest tests/ --cov=src/
```

## 📖 Dokumentasi API

### Endpoints

#### POST /predict
Melakukan prediksi diabetes berdasarkan input data kesehatan.

**Request Body:**
```json
{
  "glucose": 120,
  "blood_pressure": 80,
  "bmi": 25.5,
  "age": 35,
  "insulin": 100,
  "skin_thickness": 20,
  "diabetes_pedigree": 0.5,
  "pregnancies": 2
}
```

**Response:**
```json
{
  "prediction": "positive",
  "probability": 0.78,
  "risk_level": "high",
  "recommendations": [
    "Konsultasi dengan dokter",
    "Monitor gula darah secara rutin",
    "Jaga pola makan sehat"
  ]
}
```

#### GET /health
Health check endpoint untuk memastikan API berjalan dengan baik.

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t diabeacheck-ml .

# Run container
docker run -p 5000:5000 diabeacheck-ml
```

## 🤝 Contributing

1. Fork repository ini
2. Buat branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 Roadmap

- [x] Data collection dan preprocessing
- [x] Model development dan training
- [x] Model evaluation
- [x] API development
- [ ] Frontend integration
- [ ] Mobile app deployment
- [ ] Real-time monitoring dashboard
- [ ] Model versioning dan A/B testing

## ⚠️ Disclaimer

Model ini dikembangkan untuk tujuan edukasi dan penelitian. Hasil prediksi tidak dapat menggantikan diagnosis medis profesional. Selalu konsultasikan dengan tenaga medis yang kompeten untuk diagnosis dan pengobatan diabetes.

## 👥 Tim Pengembang

- **[Alfiah (MC796D5X0076)]** – Politeknik Baja Tegal

- **[Elaine Agustina (MC834D5X1658)]** – Universitas Pelita Harapan

- **[Rafly Ashraffi Rachmat (MC796D5Y0101)]** – Politeknik Baja Tegal

📄 Lisensi

Proyek ini dilisensikan di bawah MIT License.
MIT License

Copyright (c) 2025 DiabeaCheck Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...

## 🙏 Acknowledgments

- Terima kasih kepada [sumber dataset]
- Inspiration dari [referensi paper/project]
- Dukungan dari [institusi/mentor]

---
