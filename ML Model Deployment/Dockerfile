# dockerfile untuk DiabeaCheck FastAPI Application

# Gunakan Python 3.9 sebagai base image
FROM python:3.9

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja di dalam container
WORKDIR /app

# Salin dan install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Salin seluruh isi project ke dalam container
COPY . .

# Tambahkan user non-root
RUN adduser --disabled-password myuser
USER myuser

# Jalankan aplikasi FastAPI menggunakan Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
