from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# ✅ Replace with your actual Hugging Face Space API URL
# Example: "https://anusrii29-flower-model.hf.space/api/predict"
HF_API_URL = "https://anusrii29-flower-model.hf.space/api/predict"

# Ensure static folder for image uploads exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save image locally (optional, for preview)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Prepare the image for sending to Hugging Face API
    files = {"file": (file.filename, open(file_path, "rb"), "image/jpeg")}

    try:
        response = requests.post(HF_API_URL, files=files)
        response.raise_for_status()
        result = response.json()

        # Optional: prettify the output
        return render_template(
            'result.html',
            image_path=file_path,
            result=result
        )

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Hugging Face API request failed: {str(e)}'})
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
