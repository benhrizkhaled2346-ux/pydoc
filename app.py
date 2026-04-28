import warnings
warnings.filterwarnings("ignore")

import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import onnxruntime as ort

# =======================
# CONFIG
# =======================

CLASSES = [
    "olive_aculus_olearius", "olive_healthy", "olive_peacock_spot",
    "palm_black_scorch", "palm_fusarium_wilt", "palm_healthy",
    "palm_leaf_spots", "palm_magnesium_deficiency", "palm_manganese_deficiency",
    "palm_parlatoria_blanchardi", "palm_potassium_deficiency", "palm_rachis_blight"
]

CONFIDENCE_THRESHOLD = 0.60
MAX_IMAGE_BYTES = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fallehi_student.onnx")

# =======================
# LOAD MODEL
# =======================

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# =======================
# PREPROCESS (manual)
# =======================

def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0

    # Normalize (ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dim
    return np.expand_dims(img, axis=0).astype(np.float32)

# =======================
# SOFTMAX (no torch)
# =======================

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# =======================
# PREDICTION
# =======================

def predict(image: Image.Image):
    try:
        input_tensor = preprocess(image)

        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]

        probs = softmax(logits)

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])

        if top_prob < CONFIDENCE_THRESHOLD:
            return {"valid": False, "message": "Low confidence"}

        top3_idx = probs.argsort()[-3:][::-1]

        top3 = [
            {"class": CLASSES[i], "prob": round(float(probs[i]), 4)}
            for i in top3_idx
        ]

        return {
            "valid": True,
            "disease": CLASSES[top_idx],
            "confidence": round(top_prob * 100, 2),
            "top3": top3
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}

# =======================
# API
# =======================

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)

    if not data or "image" not in data:
        return jsonify({"error": "Missing image"}), 400

    try:
        image_bytes = base64.b64decode(data["image"])
    except:
        return jsonify({"error": "Invalid base64"}), 400

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Too large"}), 413

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 422

    result = predict(image)

    if not result.get("valid"):
        return jsonify(result), 422

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)