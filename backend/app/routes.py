from flask import Blueprint, request, jsonify, current_app as app
import numpy as np
import json
from .services.spectrum_processor import SpectrumProcessor
from .services.ml_classifier import MLClassifier
from .services.astrodash_backend import get_training_parameters, AgeBinning

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
        try:
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
        except RuntimeError as e:
            app.logger.error(f"User-friendly error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 400

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

@main.route('/api/analysis-options', methods=['GET'])
def get_analysis_options():
    try:
        try:
            pars = get_training_parameters()
            sn_types = pars['typeList']
            host_types = ['No Host'] + pars.get('galTypeList', [])
            age_binner = AgeBinning(pars['minAge'], pars['maxAge'], pars['ageBinSize'])
            age_bins = age_binner.age_labels()
        except Exception as e:
            # Fallback to mock data if model files are missing
            sn_types = ['Ia', 'Ib', 'Ic', 'II', 'IIn', 'IIb']
            age_bins = ['-10 to -5', '-5 to 0', '0 to 5', '5 to 10', '10 to 15']
            host_types = ['No Host', 'E', 'S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Irr']
        return jsonify({
            'sn_types': sn_types,
            'age_bins': age_bins,
            'host_types': host_types
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@main.route('/api/template-spectrum', methods=['GET'])
def get_template_spectrum():
    sn_type = request.args.get('sn_type', 'Ia')
    age_bin = request.args.get('age_bin', '0 to 5')
    host_type = request.args.get('host_type', 'No Host')
    try:
        import sys
        sys.path.insert(0, './astrodash')
        from astrodash.data.read_binned_templates import load_templates, get_templates, combined_sn_and_host_data
        import numpy as np
        import os
        # Path to the .npz file (relative to backend root)
        npz_path = os.path.join(os.path.dirname(__file__), '../../astrodash/astrodash/data/models_v06/models/sn_and_host_templates.npz')
        snTemplates, galTemplates = load_templates(npz_path)
        # Use a default nw, w0, w1 (should match Astrodash defaults)
        nw = 1024
        w0 = 3500
        w1 = 10000
        # Get the template info
        snInfos, snNames, hostInfos, hostNames = get_templates(sn_type, age_bin, host_type, snTemplates, galTemplates, nw)
        # Use the first sub-template (legacy GUI allows navigation, but we just use the first)
        snCoeff = 1.0
        galCoeff = 0.0 if host_type == 'No Host' else 1.0
        wave, flux, minMaxIndex = combined_sn_and_host_data(snCoeff, galCoeff, 0, snInfos[0], hostInfos[0], w0, w1, nw)
        return jsonify({
            'wave': wave.tolist(),
            'flux': flux.tolist(),
            'sn_type': sn_type,
            'age_bin': age_bin,
            'host_type': host_type
        })
    except Exception as e:
        return jsonify({'error': str(e), 'wave': [], 'flux': [], 'sn_type': sn_type, 'age_bin': age_bin, 'host_type': host_type}), 404
