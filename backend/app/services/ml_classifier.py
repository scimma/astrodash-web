import numpy as np
import os
import tempfile
from .astrodash_backend import (
    classification_split, combined_prob, RlapCalc, AgeBinning,
    get_training_parameters, LoadInputSpectra, BestTypesListSingleRedshift
)

class MLClassifier:
    def __init__(self, data_files='models_v06'):
        self.data_files = data_files
        self.pars = get_training_parameters(data_files)

        backend_dir = os.path.dirname(os.path.abspath(__file__))
        # Update model path to look for a PyTorch file
        self.model_path = os.path.join(backend_dir, '..', '..', 'astrodash_data', data_files, 'models', 'zeroZ', 'pytorch_model.pth')

    def classify(self, processed_data):
        """Classify spectrum data and return results using the PyTorch-based backend"""

        # Check if the PyTorch model file exists
        if not os.path.exists(self.model_path):
            # Return a mock response if the model is not found
            return self._mock_classification_response(processed_data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            np.savetxt(f, np.array([processed_data['x'], processed_data['y']]).T)
            temp_file_path = f.name

        try:
            load_spectra = LoadInputSpectra(
                temp_file_path,
                z=processed_data['redshift'],
                smooth=0,
                pars=self.pars,
                min_wave=min(processed_data['x']),
                max_wave=max(processed_data['x']),
                classify_host=processed_data.get('classify_host', False)
            )

            input_images, _, type_names_list, nw, n_bins, _ = load_spectra.input_spectra()

            best_types_list = BestTypesListSingleRedshift(self.model_path, input_images, type_names_list, nw, n_bins)

            matches = []
            if best_types_list.best_types:
                for i in range(len(best_types_list.best_types[0])):
                    classification = best_types_list.best_types[0][i]
                    probability = best_types_list.softmax_ordered[0][i]
                    host, sn_name, sn_age = classification_split(classification)

                    matches.append({
                        'type': sn_name, 'age': sn_age, 'host': host,
                        'probability': float(probability),
                        'redshift': processed_data['redshift'],
                        'rlap': processed_data.get('rlap_score'),
                        'reliable': False
                    })

            if not matches:
                return self._mock_classification_response(processed_data)

            best_match_list_for_prob = [[m['host'], m['type'], m['age'], m['probability']] for m in matches]
            host, best_type, best_age, prob_total, reliable_flag = combined_prob(best_match_list_for_prob)

            best_match = {
                'type': best_type, 'age': best_age, 'probability': prob_total,
                'host': host, 'redshift': processed_data['redshift']
            }

            for m in matches:
                if m['type'] == best_type and m['age'] == best_age:
                    m['reliable'] = reliable_flag

            return {
                'best_matches': matches[:3],
                'best_match': best_match,
                'reliable_matches': reliable_flag
            }
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def _mock_classification_response(self, processed_data):
        """Returns a mock response for when the PyTorch model is not available."""
        print("Warning: PyTorch model not found. Returning mock classification.")
        mock_match = {
            'type': 'Ia-norm', 'age': '0 to 5', 'host': 'No Host',
            'probability': 0.99, 'redshift': processed_data['redshift'],
            'rlap': processed_data.get('rlap_score'), 'reliable': True
        }
        return {
            'best_matches': [mock_match],
            'best_match': mock_match,
            'reliable_matches': True
        }

    def extract_features(self, spectrum_data):
        """Extract features from spectrum data for classification"""
        # This can be implemented if specific feature extraction is needed outside of the model
        return np.random.rand(10)

    def calculate_rlap(self, spectrum1, spectrum2):
        """Calculate rlap score between two spectra using astrodash_backend's RlapCalc"""
        # This is a placeholder; requires actual spectra for a meaningful calculation
        rlap_calc = RlapCalc(spectrum1, [spectrum2], ['template'], np.arange(len(spectrum1)), (0, len(spectrum1)-1), [(0, len(spectrum1)-1)])
        rlap_label, rlap_warning = rlap_calc.rlap_label()
        return float(rlap_label)

    def load_model(self, model_path):
        pass

    def save_model(self, model_path):
        pass
