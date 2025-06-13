from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
import numpy as np
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'fits', 'dat', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Basic route for testing
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "AstroDASH API is running"})

# Route to get spectrum data
@app.route('/api/spectrum', methods=['GET'])
def get_spectrum():
    # TODO: Implement spectrum data retrieval from S3
    return jsonify({"message": "Spectrum endpoint"})

# Route to process spectrum
@app.route('/api/process', methods=['POST'])
def process_spectrum():
    data = request.json
    smoothing = data.get('smoothing', 0.5)
    known_z = data.get('knownZ', False)
    z_value = data.get('zValue', 0)

    # Generate mock data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x/5) + np.random.normal(0, 0.1, 100)

    # Apply smoothing
    y_smooth = np.convolve(y, np.ones(smoothing*10)/smoothing/10, mode='same')

    return jsonify({
        'x': x.tolist(),
        'y': y_smooth.tolist(),
        'z': z_value if known_z else None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
