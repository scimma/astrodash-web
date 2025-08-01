import numpy as np
import os
import tempfile
import logging
from .astrodash_backend import (
    classification_split, combined_prob, AgeBinning,
    get_training_parameters, LoadInputSpectra, BestTypesListSingleRedshift
)
from .redshift_estimator import get_median_redshift
from .rlap_calculator import calculate_rlap_with_redshift, compute_rlap_for_matches
from .utils import prepare_log_wavelength_and_templates, get_templates_for_type_age, get_nonzero_minmax, normalize_age_bin
from .transformer_model import spectraTransformerEncoder

import torch

logger = logging.getLogger("ml_classifier")

class MLClassifier:
    def __init__(self):
        self.pars = get_training_parameters()
        logger.info("Loaded training parameters for MLClassifier.")

        services_dir = os.path.dirname(os.path.abspath(__file__))
        backend_root = os.path.join(services_dir, '..', '..')
        models_base_path = os.path.join(backend_root, 'astrodash_models')

        self.model_path_zero_z = os.path.join(models_base_path, 'zeroZ', 'pytorch_model.pth')
        self.model_path_agnostic_z = os.path.join(models_base_path, 'agnosticZ', 'pytorch_model.pth')

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

            # Debug training parameters
            logger.info(f"Training parameters keys: {list(self.pars.keys())}")
            if 'typeList' in self.pars:
                logger.info(f"Type list from training params: {self.pars['typeList']}")
            else:
                logger.warning("typeList not found in training parameters")

            input_images, _, type_names_list, nw, n_bins, minmax_indexes = load_spectra.input_spectra()

            logger.info(f"Type names list: {type_names_list}")
            logger.info(f"Number of bins: {n_bins}")

            best_types_list = BestTypesListSingleRedshift(model_path, input_images, type_names_list, nw, n_bins)

            matches = []
            if best_types_list.best_types:
                logger.info(f"Found {len(best_types_list.best_types[0])} classifications")
                for i in range(len(best_types_list.best_types[0])):
                    classification = best_types_list.best_types[0][i]
                    probability = best_types_list.softmax_ordered[0][i]
                    logger.info(f"Processing classification {i}: raw='{classification}' with probability {probability}")
                    _, sn_name, sn_age = classification_split(classification)
                    logger.info(f"Split result: sn_name='{sn_name}', sn_age='{sn_age}'")

                    matches.append({
                        'type': sn_name, 'age': sn_age,
                        'probability': float(probability),
                        'redshift': processed_data['redshift'],
                        'rlap': None,  # Will be filled below
                        'reliable': False
                    })
            else:
                logger.warning("best_types_list.best_types is empty or None")

            if not matches:
                logger.warning("No matches found. Returning mock classification.")
                return {
                    'best_matches': [],
                    'best_match': {
                        'type': 'Unknown',
                        'age': 'Unknown',
                        'probability': 0.0,
                        'redshift': processed_data['redshift'],
                        'rlap': None
                    },
                    'reliable_matches': False
                }

                        # Check if user wants RLAP calculation
            calculate_rlap = processed_data.get('calculate_rlap', False)

            # Initialize variables that might be used later
            estimated_redshift = None
            estimated_redshift_err = None

            if calculate_rlap:
                # RLap calculation using real template spectra
                logger.info("Calculating RLap score using real template spectra.")
                log_wave, input_flux_log, snTemplates, dwlog, nw, w0, w1 = prepare_log_wavelength_and_templates(processed_data)

                # Find the best match (highest probability)
                best_match_idx = 0
                if matches:
                    best_match_idx = np.argmax([m['probability'] for m in matches])
                    best_match = matches[best_match_idx]
                    sn_type = best_match['type']
                    age = best_match['age']
                    age_norm = normalize_age_bin(age)
                    logger.debug(f"Looking for template for RLap: sn_type='{sn_type}', age='{age}' (normalized: '{age_norm}')")
                    if not known_z:
                        # Redshift estimation using all templates for best match
                        template_fluxes, template_names, template_minmax_indexes = get_templates_for_type_age(snTemplates, sn_type, age_norm, log_wave)
                        if template_fluxes:
                            input_minmax_index = minmax_indexes[0] if minmax_indexes else get_nonzero_minmax(input_flux_log)
                            est_z, _, _, est_z_err = get_median_redshift(
                                input_flux_log, template_fluxes, nw, dwlog, input_minmax_index, template_minmax_indexes, template_names, outerVal=0.5
                            )
                            estimated_redshift = float(est_z) if est_z is not None else None
                            estimated_redshift_err = float(est_z_err) if est_z_err is not None else None
                        else:
                            logger.warning(f"No valid templates found for sn_type '{sn_type}' and age '{age_norm}'")
                    else:
                        template_fluxes = []
                        template_names = []
                        template_minmax_indexes = []

                if not template_fluxes:
                    logger.error("No valid templates found for RLap calculation.")
                    rlap_label, rlap_warning = "N/A", True
                    for m in matches:
                        m['rlap'] = rlap_label
                        m['rlap_warning'] = rlap_warning
                    best_match['rlap'] = rlap_label
                    best_match['rlap_warning'] = rlap_warning
                else:
                    matches, best_match = compute_rlap_for_matches(
                        matches, best_match, log_wave, input_flux_log, template_fluxes, template_names, template_minmax_indexes, known_z
                    )
                logger.info(f"RLap score calculated: {best_match.get('rlap', 'N/A')} (warning: {best_match.get('rlap_warning', False)})")
            else:
                logger.info("RLAP calculation skipped as requested by user.")
                # Set default RLAP values when calculation is skipped
                for m in matches:
                    m['rlap'] = None
                    m['rlap_warning'] = False
                best_match = matches[0] if matches else {}
                best_match['rlap'] = None
                best_match['rlap_warning'] = False

            best_match_list_for_prob = [[m['type'], m['age'], m['probability']] for m in matches]
            best_type, best_age, prob_total, reliable_flag = combined_prob(best_match_list_for_prob)

            best_match = {
                'type': best_type, 'age': best_age, 'probability': prob_total,
                'redshift': processed_data['redshift'],
                'rlap': best_match['rlap']
            }

            for m in matches:
                if m['type'] == best_type and m['age'] == best_age:
                    m['reliable'] = reliable_flag

            # Attach estimated redshift to best_match and best_matches if available
            if not known_z and estimated_redshift is not None:
                best_match['estimated_redshift'] = estimated_redshift
                best_match['estimated_redshift_error'] = estimated_redshift_err
                for m in matches:
                    m['estimated_redshift'] = estimated_redshift
                    m['estimated_redshift_error'] = estimated_redshift_err

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

    def _mock_classification_response(self, processed_data, model_path):
        """Return a mock classification response when model is not available"""
        logger.warning("Returning mock classification response")
        redshift = processed_data.get('redshift', 0.0)
        return {
            'best_matches': [],
            'best_match': {},
            'reliable_matches': False
        }

    def load_model_from_state_dict(self, state_dict, n_classes):
        from app.services.astrodash_backend import AstroDashPyTorchNet
        # Use im_width=32 as in the default model
        model = AstroDashPyTorchNet(n_types=n_classes, im_width=32)
        model.load_state_dict(state_dict)
        model.eval()
        return model

def _interpolate_to_1024(arr):
    arr = np.asarray(arr)
    if len(arr) == 1024:
        return arr
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, 1024)
    return np.interp(x_new, x_old, arr)

class TransformerClassifier:
    def __init__(self):
        services_dir = os.path.dirname(os.path.abspath(__file__))
        backend_root = os.path.join(services_dir, '..', '..')
        self.model_path = os.path.join(backend_root, 'astrodash_models', 'yuqing_models', 'TF_wiserep_v6.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if model file exists
        if not os.path.exists(self.model_path):
            logger.warning(f"Transformer model file not found at {self.model_path}. Will use mock responses.")
            self.model = None
        else:
            self.model = self._load_model()
            logger.info(f"Loaded Transformer model from {self.model_path}")

        # Updated label mapping for 5 classes
        self.label_mapping = {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        logger.info(f"Using label mapping: {self.label_mapping}")

    def _load_model(self):
        """Load the transformer model with proper initialization"""
        try:
            # Model hyperparameters - adjusted to match the saved model
            bottleneck_length = 1  # Changed from 8 to 1 based on saved model
            model_dim = 128
            num_heads = 4
            num_layers = 6  # Changed from 4 to 6 based on saved model (has blocks 0-5)
            num_classes = 5  # 5 classes: Ia, IIn, SLSNe-I, II, Ib/c
            ff_dim = 256
            dropout = 0.1
            selfattn = False

            # Initialize the model
            model = spectraTransformerEncoder(
                bottleneck_length=bottleneck_length,
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_classes=num_classes,
                ff_dim=ff_dim,
                dropout=dropout,
                selfattn=selfattn
            ).to(self.device)

            # Load the state dict
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            logger.info(f"Transformer model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
            return model

        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            raise

    def classify(self, processed_data):
        """
        Classify spectrum data using the transformer-based model.
        processed_data: dict with keys 'x' (wavelength), 'y' (flux), 'redshift'
        Returns: dict with classification results.
        """
        try:
            # If model is not available, return mock response
            if self.model is None:
                logger.warning("Transformer model not available, returning mock classification.")
                return self._mock_classification_response(processed_data)

            # Extract wavelength, flux, and redshift from processed data
            wavelength_data = _interpolate_to_1024(processed_data['x'])  # wavelength
            flux_data = _interpolate_to_1024(processed_data['y'])        # flux
            redshift = processed_data.get('redshift', 0.0)

            # Convert to tensors with proper shape
            wavelength = torch.tensor(wavelength_data, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 1024]
            flux = torch.tensor(flux_data, dtype=torch.float32).unsqueeze(0).to(self.device)              # [1, 1024]
            redshift_tensor = torch.tensor([[redshift]], dtype=torch.float32).to(self.device)             # [1, 1]

            logger.info(f"Input shapes - wavelength: {wavelength.shape}, flux: {flux.shape}, redshift: {redshift_tensor.shape}")
            logger.info(f"Redshift value: {redshift}")

            with torch.no_grad():
                logits = self.model(wavelength, flux, redshift_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            logger.info(f"Model output probabilities: {probs}")

            # Get top 3 predictions using the correct label mapping
            top_indices = np.argsort(probs)[::-1][:3]
            matches = []

            # Check if user wants RLAP calculation (note: transformer model doesn't support RLAP calculation)
            calculate_rlap = processed_data.get('calculate_rlap', False)
            if calculate_rlap:
                logger.info("RLAP calculation requested but not supported by transformer model. Setting RLAP to None.")

            for idx in top_indices:
                class_name = self.idx_to_label.get(idx, f'unknown_class_{idx}')
                matches.append({
                    'type': class_name,
                    'probability': float(probs[idx]),
                    'redshift': redshift,
                    'rlap': None,  # Not calculated for transformer model (RLAP requires template matching)
                    'reliable': probs[idx] > 0.5  # Simple reliability threshold
                })

            best_match = matches[0] if matches else {}

            logger.info(f"Classification results - best match: {best_match}")

            return {
                'best_matches': matches,
                'best_match': best_match,
                'reliable_matches': best_match.get('reliable', False) if best_match else False
            }
        except Exception as e:
            logger.error(f"Error during transformer classification: {e}")
            # Return mock response on error
            return self._mock_classification_response(processed_data)

    def _mock_classification_response(self, processed_data):
        """Return a mock classification response when model is not available"""
        logger.warning("Returning mock transformer classification response")
        redshift = processed_data.get('redshift', 0.0)

        # Mock response with transformer model's 5 classes
        mock_matches = [
            {
                'type': 'Ia',
                'probability': 0.85,
                'redshift': redshift,
                'rlap': None,
                'reliable': True
            },
            {
                'type': 'II',
                'probability': 0.10,
                'redshift': redshift,
                'rlap': None,
                'reliable': False
            },
            {
                'type': 'IIn',
                'probability': 0.05,
                'redshift': redshift,
                'rlap': None,
                'reliable': False
            }
        ]

        return {
            'best_matches': mock_matches,
            'best_match': mock_matches[0],
            'reliable_matches': True
        }
