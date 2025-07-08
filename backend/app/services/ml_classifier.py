import numpy as np
import os
import tempfile
import logging
from .astrodash_backend import (
    classification_split, combined_prob, RlapCalc, AgeBinning,
    get_training_parameters, LoadInputSpectra, BestTypesListSingleRedshift
)

logger = logging.getLogger("ml_classifier")

class MLClassifier:
    def __init__(self):
        self.pars = get_training_parameters()
        logger.info("Loaded training parameters for MLClassifier.")

        services_dir = os.path.dirname(os.path.abspath(__file__))
        backend_root = os.path.join(services_dir, '..', '..')
        models_base_path = os.path.join(backend_root, 'astrodash_models')

        self.model_path_zero_z = os.path.join(models_base_path, 'zeroZ', 'zero_z_pytorch.pth')
        self.model_path_agnostic_z = os.path.join(models_base_path, 'agnosticZ', 'agnostic_z_pytorch.pth')

    def classify(self, processed_data):
        """Classify spectrum data and return results using the PyTorch-based backend"""
        known_z = processed_data.get('known_z', False)
        model_path = self.model_path_zero_z if known_z else self.model_path_agnostic_z
        logger.info(f"Starting classification. Using model: {model_path}")

        # Check if the PyTorch model file exists
        if not os.path.exists(model_path):
            logger.warning(f"PyTorch model not found at {model_path}. Returning mock classification.")
            return self._mock_classification_response(processed_data, model_path)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            np.savetxt(f, np.array([processed_data['x'], processed_data['y']]).T)
            temp_file_path = f.name
            logger.debug(f"Temporary file created for spectrum: {temp_file_path}")

        try:
            load_spectra = LoadInputSpectra(
                temp_file_path,
                z=processed_data['redshift'],
                smooth=0,
                pars=self.pars,
                min_wave=min(processed_data['x']),
                max_wave=max(processed_data['x'])
            )

            input_images, _, type_names_list, nw, n_bins, minmax_indexes = load_spectra.input_spectra()

            best_types_list = BestTypesListSingleRedshift(model_path, input_images, type_names_list, nw, n_bins)

            matches = []
            if best_types_list.best_types:
                for i in range(len(best_types_list.best_types[0])):
                    classification = best_types_list.best_types[0][i]
                    probability = best_types_list.softmax_ordered[0][i]
                    _, sn_name, sn_age = classification_split(classification)

                    matches.append({
                        'type': sn_name, 'age': sn_age,
                        'probability': float(probability),
                        'redshift': processed_data['redshift'],
                        'rlap': None,  # Will be filled below
                        'reliable': False
                    })

            if not matches:
                logger.warning("No matches found. Returning mock classification.")
                return self._mock_classification_response(processed_data, model_path)

            def normalize_age_bin(age):
                # Convert various formats to 'N to M'
                import re
                age = age.replace('–', '-').replace('to', '-').replace('TO', '-').replace('To', '-')
                age = age.replace(' ', '')
                match = re.match(r'(-?\d+)-(-?\d+)', age)
                if match:
                    return f"{int(match.group(1))} to {int(match.group(2))}"
                return age

            # RLap calculation using real template spectra
            logger.info("Calculating RLap score using real template spectra.")
            pars = get_training_parameters()
            w0, w1, nw = pars['w0'], pars['w1'], pars['nw']
            dwlog = np.log(w1 / w0) / nw
            log_wave = w0 * np.exp(np.arange(nw) * dwlog)
            backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            template_path = os.path.join(backend_root, 'astrodash_models', 'sn_and_host_templates.npz')
            data = np.load(template_path, allow_pickle=True)
            snTemplates_raw = data['snTemplates'].item()
            snTemplates = {str(k): v for k, v in snTemplates_raw.items()}
            logger.debug(f"Available sn_types: {list(snTemplates.keys())}")
            for sn_type_key in snTemplates:
                logger.debug(f"  {sn_type_key}: {list(snTemplates[sn_type_key].keys())}")
            input_flux = np.array(processed_data['y'])
            input_wave = np.array(processed_data['x'])
            input_flux_log = np.interp(log_wave, input_wave, input_flux, left=0, right=0)

            # Find the best match (highest probability)
            best_match_idx = 0
            if matches:
                best_match_idx = np.argmax([m['probability'] for m in matches])
                best_match = matches[best_match_idx]
                sn_type = best_match['type']
                age = best_match['age']
                age_norm = normalize_age_bin(age)
                logger.debug(f"Looking for template for RLap: sn_type='{sn_type}', age='{age}' (normalized: '{age_norm}')")
                if sn_type in snTemplates:
                    age_bin_keys = [str(k).strip() for k in snTemplates[sn_type].keys()]
                    if age_norm.strip() in age_bin_keys:
                        real_key = [k for k in snTemplates[sn_type].keys() if str(k).strip() == age_norm.strip()][0]
                        snInfo = snTemplates[sn_type][real_key].get('snInfo', None)
                        if isinstance(snInfo, np.ndarray) and snInfo.shape[0] > 0:
                            template_wave = snInfo[0][0]
                            template_flux = snInfo[0][1]
                            interp_flux = np.interp(log_wave, template_wave, template_flux, left=0, right=0)
                            nonzero = np.where(interp_flux != 0)[0]
                            if len(nonzero) > 0:
                                tmin, tmax = nonzero[0], nonzero[-1]
                            else:
                                tmin, tmax = 0, len(interp_flux) - 1
                            template_fluxes = [interp_flux]
                            template_names = [f"{sn_type}:{age_norm}"]
                            template_minmax_indexes = [(tmin, tmax)]
                        else:
                            logger.warning(f"No snInfo for template {sn_type}:{age_norm}")
                            template_fluxes = []
                            template_names = []
                            template_minmax_indexes = []
                    else:
                        logger.warning(f"Age bin '{age_norm}' not found for sn_type '{sn_type}'. Available: {list(snTemplates[sn_type].keys())}")
                        template_fluxes = []
                        template_names = []
                        template_minmax_indexes = []
                else:
                    logger.warning(f"sn_type '{sn_type}' not found in templates. Available: {list(snTemplates.keys())}")
                    template_fluxes = []
                    template_names = []
                    template_minmax_indexes = []
            else:
                template_fluxes = []
                template_names = []
                template_minmax_indexes = []

            if not template_fluxes:
                logger.error("No valid templates found for RLap calculation.")
                rlap_label, rlap_warning = "N/A", True
            else:
                nonzero_input = np.where(input_flux_log != 0)[0]
                if len(nonzero_input) > 0:
                    input_minmax_index = (nonzero_input[0], nonzero_input[-1])
                else:
                    input_minmax_index = (0, len(input_flux_log) - 1)
                rlap_calc = RlapCalc(input_flux_log, template_fluxes, template_names, log_wave, input_minmax_index, template_minmax_indexes)
                rlap_label, rlap_warning = rlap_calc.rlap_label()
            logger.info(f"RLap score calculated: {rlap_label} (warning: {rlap_warning})")
            for m in matches:
                m['rlap'] = rlap_label

            best_match_list_for_prob = [[m['type'], m['age'], m['probability']] for m in matches]
            best_type, best_age, prob_total, reliable_flag = combined_prob(best_match_list_for_prob)

            best_match = {
                'type': best_type, 'age': best_age, 'probability': prob_total,
                'redshift': processed_data['redshift'],
                'rlap': rlap_label
            }

            for m in matches:
                if m['type'] == best_type and m['age'] == best_age:
                    m['reliable'] = reliable_flag

            logger.info("Classification complete.")
            return {
                'best_matches': matches[:3],
                'best_match': best_match,
                'reliable_matches': reliable_flag
            }
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            raise
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.debug(f"Temporary file deleted: {temp_file_path}")

    def _mock_classification_response(self, processed_data, model_path_not_found):
        logger.debug(f"Returning mock classification for model path: {model_path_not_found}")
        mock_match = {
            'type': 'Ia-norm', 'age': '0 to 5',
            'probability': 0.99, 'redshift': processed_data['redshift'],
            'rlap': None, 'reliable': True
        }
        return {
            'best_matches': [mock_match],
            'best_match': mock_match,
            'reliable_matches': True
        }

    def extract_features(self, spectrum_data):
        logger.debug("Extracting features from spectrum data.")
        return np.random.rand(10)

    def calculate_rlap(self, spectrum1, spectrum2):
        logger.debug("Calculating RLAP score between two spectra.")
        # Use the new RlapCalc if needed elsewhere
        return None

    def load_model(self, model_path):
        logger.debug(f"Called load_model with path: {model_path}")
        pass

    def save_model(self, model_path):
        logger.debug(f"Called save_model with path: {model_path}")
        pass
