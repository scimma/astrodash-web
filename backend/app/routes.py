from flask import Blueprint, request, jsonify, current_app as app
import numpy as np
import json
from .services.spectrum_processor import SpectrumProcessor
from .services.ml_classifier import MLClassifier

main = Blueprint('main', __name__)
spectrum_processor = SpectrumProcessor()
ml_classifier = MLClassifier()

@main.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        "message": "Welcome to Astrodash API",
        "endpoints": {
            "health": "/health",
            "process": "/process",
            "upload": "/upload",
            "osc-references": "/api/osc-references"
        }
    })

@main.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@main.route('/api/osc-references', methods=['GET'])
def get_osc_references():
    """Get a list of available OSC references."""
    try:
        # For now, return a hardcoded list for testing since the data directory might not exist
        available_refs = [
            'osc-sn2002er-0',
            'osc-sn2011fe-0',
            'osc-sn2014J-0',
            'osc-sn1998aq-0',
            'osc-sn1999ee-0',
            'osc-sn2005cf-0',
            'osc-sn2007af-0',
            'osc-sn2009ig-0',
            'osc-sn2012cg-0',
            'osc-sn2013dy-0',
            'osc-sn2014dt-0',
            'osc-sn2016coj-0',
        ]
        app.logger.info(f"Returning {len(available_refs)} OSC references.")
        return jsonify({
            'status': 'success',
            'references': available_refs
        })
    except Exception as e:
        app.logger.error(f"Error fetching OSC references: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@main.route('/process', methods=['POST'])
def process_spectrum():
    app.logger.info("Received request to /process")
    try:
        # Get parameters from request
        params_str = request.form.get('params')
        app.logger.info(f"Raw params string: {params_str}")

        if params_str:
            params = json.loads(params_str)
        else:
            params = {}

        app.logger.info(f"Parsed params: {params}")

        # Determine source of spectrum data (file or OSC reference)
        spectrum_data = None
        if 'file' in request.files:
            file = request.files['file']
            if file:
                app.logger.info(f"Received file: {file.filename}")
                spectrum_data = spectrum_processor.read_file(file)
        elif 'oscRef' in params:
            osc_ref = params.get('oscRef')
            if osc_ref:
                app.logger.info(f"Received OSC reference: {osc_ref}")
                spectrum_data = spectrum_processor.read_file(osc_ref)

        if not spectrum_data:
            return jsonify({"error": "No spectrum file or OSC reference provided"}), 400

        # Process spectrum using x and y data
        processed_data = spectrum_processor.process(
            x=spectrum_data['x'],
            y=spectrum_data['y'],
            smoothing=params.get('smoothing', 0),
            known_z=params.get('knownZ', False),
            z_value=params.get('zValue'),
            min_wave=params.get('minWave'),
            max_wave=params.get('maxWave'),
            classify_host=params.get('classifyHost', False),
            calculate_rlap=params.get('calculateRlap', False)
        )

        # Classify spectrum
        classification_results = ml_classifier.classify(processed_data)

        return jsonify({
            "spectrum": processed_data,
            "classification": classification_results
        })

    except Exception as e:
        app.logger.error(f"Error in process_spectrum: {e}", exc_info=True) # Log full traceback
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@main.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Process the uploaded file
        spectrum_data = spectrum_processor.read_file(file)
        return jsonify(spectrum_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
