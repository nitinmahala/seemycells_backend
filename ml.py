from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import torch

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# Load model once during app startup
try:
    model = YOLO("best.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        img = Image.open(file.stream).convert("RGB")
        results = model.predict(img)
        boxes = results[0].boxes

        if boxes is not None and boxes.data.numel() > 0:
            class_ids = boxes.data[:, 5].int().tolist()
            class_names = [model.names[i] for i in class_ids]
        else:
            class_names = []

        count = Counter(class_names)

        # Return class-wise count
        response = {
            "WBC": count.get("WBC", 0),
            "RBC": count.get("RBC", 0),
            "Platelets": count.get("platelets", 0)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Failed to process image. {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
