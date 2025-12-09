from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow import keras

MODEL_PATH = "emg_ann.keras"
SCALER_PATH = "scaler.joblib"

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded OK")

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded OK")

CLASS_LABELS = {
    0: "Fist",
    1: "Wrist Extension",
    2: "Wrist Flexion"
}

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: list[float]

app = FastAPI(title="EMG ANN Service")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = np.array(req.features, dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)
    probs = model.predict(x_scaled, verbose=0)[0]

    class_id = int(np.argmax(probs))
    return PredictResponse(
        class_id=class_id,
        class_name=CLASS_LABELS[class_id],
        probabilities=probs.tolist()
    )
