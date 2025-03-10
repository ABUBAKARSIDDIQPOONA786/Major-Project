from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from utils.qr_validator import parse_upi_qr, predict_qr_code_safety

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/qr-validator", methods=["GET", "POST"])
def qr_validator():
    if request.method == "POST":
        if "qr_code" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        qr_file = request.files["qr_code"]
        if qr_file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(qr_file.filename):
            return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed."}), 400

        filename = secure_filename(qr_file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        qr_file.save(file_path)

        try:
            # Extract QR Code Data
            qr_data = parse_upi_qr(file_path)
            if not qr_data:
                return jsonify({"error": "Invalid QR Code"}), 400

            # Get QR Safety Prediction
            safety_result = predict_qr_code_safety(qr_data)

            return jsonify({"status": "success", "report": safety_result})
        except Exception as e:
            logging.error(f"Error processing QR Code: {str(e)}")
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return render_template("qr_validator.html")

if __name__ == "__main__":
    app.run(debug=True)