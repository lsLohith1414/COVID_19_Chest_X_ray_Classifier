
import io
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input

# Simple in-memory model cache to avoid reloading model on every request
_MODEL_CACHE = {}

def load_model_cached(model_path: str):
    """Load a Keras model once and reuse it (simple cache)."""
    global _MODEL_CACHE
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = load_model(model_path)
    return _MODEL_CACHE[model_path]

def _load_image_from_bytes(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """
    Read image bytes, convert to RGB, resize to target_size, and return a batched numpy array.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32")
    # shape -> (H, W, 3); convert to (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(
    model_path: str,
    image_bytes: bytes,
    class_names: List[str],
    target_size=(224, 224)
) -> Dict[str, Any]:

    # 1) load model (cached)
    model = load_model_cached(model_path)

    # 2) prepare image
    arr = _load_image_from_bytes(image_bytes, target_size=target_size)

    # 3) preprocess (VGG19)
    x = preprocess_input(arr)

    # 4) predict
    preds = model.predict(x)  # shape (1, C)
    probs = preds[0]
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    # 5) build probabilities dict
    probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return {
        "predicted_class": pred_name,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": probabilities
    }
