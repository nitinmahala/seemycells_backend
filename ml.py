from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
from collections import Counter

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream)
    results = model.predict(img)
    boxes = results[0].boxes.data
    class_names = [model.names[int(cls)] for cls in boxes[:, 5]]

    count = Counter(class_names)

    # Convert to regular dict and rename keys to match frontend
    response = {
        "WBC": count.get("WBC", 0),
        "RBC": count.get("RBC", 0),
        "Platelets": count.get("platelets", 0)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
