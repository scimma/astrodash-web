from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from services.spectrum_processor import SpectrumProcessor
from services.classifier import SupernovaClassifier

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Initialize services
    spectrum_processor = SpectrumProcessor()
    classifier = SupernovaClassifier()

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy'})

    @app.route('/api/osc-references', methods=['GET'])
    def get_osc_references():
        """Get a list of available OSC references."""
        try:
            # Get the list of available spectra from the data directory
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            available_refs = []

            # Walk through the data directory
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.dat'):
                        # Extract the OSC reference from the filename
                        # Assuming filename format: osc-snXXXXXX-X.dat
                        ref = file.replace('.dat', '')
                        available_refs.append(ref)

            return jsonify({
                'status': 'success',
                'references': available_refs
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.route('/process', methods=['POST'])
    def process_spectrum():
        try:
            # Get parameters from request
            params = request.form.get('params')
            if params:
                params = json.loads(params)
            else:
                params = {}

            # Get file from request if present
            file = request.files.get('file')

            # Process spectrum
            result = spectrum_processor.process(
                file=file,
                osc_ref=params.get('oscRef'),
                smoothing=params.get('smoothing', 0),
                known_z=params.get('knownZ', False),
                z_value=params.get('zValue'),
                min_wave=params.get('minWave'),
                max_wave=params.get('maxWave'),
                classify_host=params.get('classifyHost', False),
                calculate_rlap=params.get('calculateRlap', False)
            )

            return jsonify(result)

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    return app
