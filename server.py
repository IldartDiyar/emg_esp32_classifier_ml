from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "emg_ann.keras"
SCALER_PATH = "scaler.joblib"

print("Loading model (safe mode)...")

try:
    # Try normal load first
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded normally.")
except Exception as e:
    print("Normal load failed:", e)
    print("Attempting safe rebuild...")

    # Load model without compiling, allowing legacy configs
    loaded = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False  # Disable keras3 strict mode
    )

    # Rebuild a NEW Sequential model for TF 2.x compatibility
    new_model = keras.Sequential()

    for layer in loaded.layers:
        if isinstance(layer, keras.layers.InputLayer):
            # Rebuild the InputLayer without batch_shape
            new_model.add(
                keras.layers.Input(shape=layer.input_shape[1:])
            )
        else:
            new_model.add(layer)

    # Explicitly build for compatibility
    new_model.build(input_shape=(None, loaded.input_shape[1]))

    model = new_model
    print("Model rebuilt successfully for TF2.x.")

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
