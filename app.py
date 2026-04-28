import warnings
warnings.filterwarnings("ignore")

import os
import io
import base64
import numpy as np
import torch  # only used for softmax
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms
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
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB (fixed from your mismatch)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "fallehi_student.onnx")

# =======================
# LOAD ONNX MODEL
# =======================

session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]  # change to CUDAExecutionProvider if GPU
)

input_name = session.get_inputs()[0].name

# =======================
# IMAGE TRANSFORM
# =======================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# =======================
# PREDICTION
# =======================

def predict(image: Image.Image) -> dict:
    try:
        tensor = transform(image).unsqueeze(0)
        np_input = tensor.numpy().astype(np.float32)

        outputs = session.run(None, {input_name: np_input})
        logits = outputs[0][0]

        probs = torch.softmax(torch.tensor(logits), dim=0)

        top_prob = float(probs.max())
        top_idx = int(probs.argmax())

        if top_prob < CONFIDENCE_THRESHOLD:
            return {"valid": False, "message": "Not a leaf or confidence too low"}

        top3 = sorted(
            [{"class": CLASSES[i], "prob": round(float(probs[i]), 4)}
             for i in range(len(CLASSES))],
            key=lambda x: x["prob"], reverse=True
        )[:3]

        return {
            "valid": True,
            "disease": CLASSES[top_idx],
            "confidence": round(top_prob * 100, 2),
            "top3": top3,
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}

# =======================
# FLASK APP
# =======================

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "onnx",
        "providers": session.get_providers()
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)

    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field (base64)"}), 400

    raw = data["image"]

    if not isinstance(raw, str):
        return jsonify({"error": "'image' must be base64 string"}), 400

    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except Exception:
        return jsonify({"error": "Invalid base64"}), 400

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image exceeds 10MB"}), 413

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 422

    result = predict(image)

    if not result.get("valid", False):
        return jsonify(result), 422

    return jsonify(result)

# =======================
# ENTRYPOINT
# =======================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)