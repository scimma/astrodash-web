import os
import torch
import numpy as np
import pickle
from typing import Any, Optional
from app.infrastructure.ml.classifiers.base import BaseClassifier
from app.infrastructure.ml.processors.data_processor import DashSpectrumProcessor
from app.infrastructure.ml.classifiers.architectures import AstroDashPyTorchNet
import logging
from app.config.settings import get_settings, Settings
from app.infrastructure.ml.dash_utils import combined_prob

logger = logging.getLogger("dash_classifier")

class DashClassifier(BaseClassifier):
    """
    Production-grade Dash (CNN) classifier for supernova spectra.
    Uses dependency injection for processor and config.
    """
    def __init__(self, config: Settings = None, processor: Optional[DashSpectrumProcessor] = None):
        super().__init__(config)
        self.config = config or get_settings()
        self.processor = processor
        self.model = None
        self.model_path = self.config.dash_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self.nw = getattr(self.config, "nw", 1024)
        self.w0 = getattr(self.config, "w0", 3500.0)
        self.w1 = getattr(self.config, "w1", 10000.0)
        if not self.processor:
            self.processor = DashSpectrumProcessor(self.w0, self.w1, self.nw)

        # Load type names from training parameters like old backend
        self.type_names_list = self._load_type_names()

    def _load_type_names(self):
        """Load type names from training parameters file like the old backend."""
        try:
            # Load training parameters from the same location as old backend
            services_dir = os.path.dirname(os.path.abspath(__file__))
            backend_root = os.path.join(services_dir, '..', '..', '..', '..', '..')
            models_dir = os.path.join(backend_root, 'backend', 'astrodash_models')
            training_params_path = os.path.join(models_dir, "zeroZ/training_params.pickle")

            with open(training_params_path, 'rb') as f:
                pars = pickle.load(f, encoding='latin1')

            # Extract type list and generate type names like old backend
            type_list = pars.get('typeList', [])
            min_age = pars.get('minAge', -5)
            max_age = pars.get('maxAge', 15)
            age_bin_size = pars.get('ageBinSize', 4)

            # Generate age labels like old backend
            age_labels = []
            age_bin_prev = 0
            age_label_min = min_age
            for age in np.arange(min_age, max_age, 0.5):
                age_bin = int(round(age / age_bin_size)) - int(round(min_age / age_bin_size))
                if age_bin != age_bin_prev:
                    age_label_max = int(round(age))
                    age_labels.append(f"{int(age_label_min)} to {age_label_max}")
                    age_label_min = age_label_max
                age_bin_prev = age_bin
            age_labels.append(f"{int(age_label_min)} to {int(max_age)}")

            # Generate type names like old backend
            type_names = []
            for t_type in type_list:
                for age_label in age_labels:
                    type_names.append(f"{t_type}: {age_label}")

            logger.info(f"Loaded {len(type_names)} type names from training parameters")
            return type_names

        except Exception as e:
            logger.error(f"Failed to load type names from training parameters: {e}")
            return []

    def _classification_split(self, classification_string):
        """Split classification string like 'Ia: 2 to 6' into type and age."""
        parts = classification_string.split(': ')
        return "", parts[0], parts[1]

    def _default_model_path(self):
        return self.config.dash_model_path

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Dash model not found at {self.model_path}. Classifier will not work.")
            self.model = None
            return
        state_dict = torch.load(self.model_path, map_location=self.device)
        n_types = state_dict['classifier.3.weight'].shape[0]
        self.model = self.load_model_from_state_dict(state_dict, n_types)
        logger.info(f"Dash model loaded from {self.model_path}")

    async def classify(self, spectrum: Any) -> dict:
        """
        Preprocess the spectrum, run inference, and return classification results.
        """
        if self.model is None:
            logger.error("Dash model is not loaded. Returning empty result.")
            return {}
        # Assume spectrum.x, spectrum.y, spectrum.redshift
        x = np.array(spectrum.x)
        y = np.array(spectrum.y)
        z = getattr(spectrum, 'redshift', 0.0) or 0.0
        # Preprocess
        processed_flux, min_idx, max_idx, z = self.processor.process(x, y, z)
        input_tensor = torch.from_numpy(processed_flux).float().reshape(1, -1)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        softmax = outputs[0].cpu().numpy()

        # Only use the first n_bins outputs (corresponding to actual galaxy types)
        # The model may have been trained with more classes but only the first n_bins are valid
        n_bins = len(self.type_names_list)
        softmax = softmax[:n_bins]

        logger.info(f"Softmax shape: {softmax.shape}, type_names_list length: {len(self.type_names_list)}")
        logger.info(f"Using first {n_bins} outputs from model")

        # Process ALL classifications like the old backend (not just top 3)
        # This is crucial for scientific accuracy in probability combination
        all_indices = np.argsort(softmax)[::-1]  # Sort all indices, not just top 3
        matches = []

        logger.info(f"Processing all {len(all_indices)} classifications")

        for idx in all_indices:
            if idx < len(self.type_names_list):
                classification = self.type_names_list[idx]
                _, sn_type, sn_age = self._classification_split(classification)
                matches.append({
                    'type': sn_type,
                    'age': sn_age,
                    'probability': float(softmax[idx]),
                    'redshift': z,
                    'rlap': None,
                    'reliable': False
                })
                logger.info(f"Added match {len(matches)}: {sn_type} ({sn_age}) - {float(softmax[idx]):.3f}")
            else:
                logger.warning(f"Index {idx} out of range for type_names_list (length: {len(self.type_names_list)})")

        if not matches:
            logger.warning("No valid matches found. Returning mock classification.")
            return {
                'best_matches': [],
                'best_match': {
                    'type': 'Unknown',
                    'age': 'Unknown',
                    'probability': 0.0,
                    'redshift': z,
                    'rlap': None
                },
                'reliable_matches': False
            }

        # Use combined_prob like the original DASH package with ALL matches
        best_match_list_for_prob = [[m['type'], m['age'], m['probability']] for m in matches]
        best_type, best_age, prob_total, reliable_flag = combined_prob(best_match_list_for_prob)

        # Update the best match with combined probability
        best_match = {
            'type': best_type,
            'age': best_age,
            'probability': prob_total,
            'redshift': z,
            'rlap': None,
            'reliable': reliable_flag
        }

        # Update matches to mark the best one as reliable
        for m in matches:
            if m['type'] == best_type and m['age'] == best_age:
                m['reliable'] = reliable_flag

        # Return only top 3 matches for display, but use all for probability calculation
        return {
            'best_matches': matches[:3],  # Only return top 3 for display
            'best_match': best_match,
            'reliable_matches': reliable_flag
        }

    def load_model_from_state_dict(self, state_dict, n_classes):
        """
        Load a model from a state dict with the specified number of classes.

        Args:
            state_dict: PyTorch state dict containing model weights
            n_classes: Number of output classes for the model

        Returns:
            AstroDashPyTorchNet: Loaded and configured model
        """
        # Use im_width=32 as in the default model
        model = AstroDashPyTorchNet(n_types=n_classes, im_width=32)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def update_model_from_state_dict(self, state_dict, n_classes):
        """
        Update the current model with a new state dict.

        Args:
            state_dict: PyTorch state dict containing model weights
            n_classes: Number of output classes for the model
        """
        self.model = self.load_model_from_state_dict(state_dict, n_classes)
        logger.info(f"Dash model updated with new state dict (n_classes={n_classes})")
