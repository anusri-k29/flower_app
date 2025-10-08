from flask import Flask, request, render_template
import requests
import os
import json
import base64
import boto3
from datetime import datetime

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# HF Spaces API endpoint
HF_API_URL = "https://anusrii29-flower-model.hf.space/predict"

# Class names JSON URL from Hugging Face
CLASS_NAMES_URL = "https://huggingface.co/anusrii29/plant_id/resolve/main/class_names.json"
CLASS_NAMES = None

# AWS S3 Configuration
S3_BUCKET = "flower-id-uploads"
S3_ACCESS_KEY = "xyz"

S3_SECRET_KEY = "xyx"
S3_REGION = "eu-north-1"

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION
)

# Load class names dynamically from Hugging Face
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

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Upload to S3
    try:
        s3.upload_file(file_path, S3_BUCKET, file.filename)
        s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{file.filename}"
    except Exception as e:
        return {"error": f"Failed to upload to S3: {str(e)}"}

    # Send file to Hugging Face API
    files = {"file": (file.filename, open(file_path, "rb"), "image/jpeg")}
    try:
        response = requests.post(HF_API_URL, files=files)
        response.raise_for_status()
        result = response.json()

        top_flower = result.get("flowerName", "Unknown")
        confidence = result.get("confidence", 0.0)

        # Map prediction to class names if available
        classes = load_class_names()
        if isinstance(classes, dict) and "flower" in classes:
            classes = classes["flower"]

        if classes and str(top_flower).isdigit():
            top_flower = classes[int(top_flower)]

        # Save prediction result JSON to S3
        prediction_data = {
            "filename": file.filename,
            "flower": top_flower,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        prediction_filename = f"predictions/{file.filename.split('.')[0]}_result.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=prediction_filename,
            Body=json.dumps(prediction_data),
            ContentType="application/json"
        )

        return render_template("result.html", image_path=s3_url, result=top_flower, confidence=confidence)

    except requests.exceptions.RequestException as e:
        return {"error": f"Hugging Face API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
