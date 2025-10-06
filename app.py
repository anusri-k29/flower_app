from flask import Flask, request, render_template
import requests
import os
import json
import base64

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# HF Spaces API endpoint
HF_API_URL = "https://anusrii29-flower-model.hf.space/predict"

# Class names JSON URL from Hugging Face
CLASS_NAMES_URL = "https://huggingface.co/anusrii29/plant_id/resolve/main/class_names.json"
CLASS_NAMES = None

# Load class names dynamically from HF
def load_class_names():
    global CLASS_NAMES
    if CLASS_NAMES is not None:
        return CLASS_NAMES
    try:
        resp = requests.get(CLASS_NAMES_URL)
        resp.raise_for_status()
        CLASS_NAMES = resp.json()
        return CLASS_NAMES
    except Exception as e:
        print(f"Failed to load class names: {e}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files['file']
    if file.filename == '':
        return {"error": "No selected file"}, 400

    # Save image temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Convert image to base64 for HF Spaces API
    with open(file_path, "rb") as f:
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    payload = {"data": [img_b64]}

    try:
        response = requests.post(HF_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        # HF Spaces returns {"flowerName": "...", "confidence": ..., "predictions": [...]}
        top_flower = result.get("flowerName", "Unknown")
        confidence = result.get("confidence", 0.0)

        # Load class names from HF if not loaded
        classes = load_class_names()
        if isinstance(classes, dict) and "flower" in classes:
            classes = classes["flower"]  # handle dict structure

        # Map prediction to class name if possible
        if classes and top_flower.isdigit():
            top_flower = classes[int(top_flower)]

        # Render result page
        return render_template("result.html", image_path=file_path, result=top_flower, confidence=confidence)

    except requests.exceptions.RequestException as e:
        return {"error": f"Hugging Face API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
