FROM python:3.10-bullseye

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    liblapack-dev \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server.py .
COPY emg_ann.keras .
COPY scaler.joblib .
COPY requirements.txt .

RUN pip install --no-cache-dir tensorflow-cpu==2.13.0

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
