import os
import torch
import numpy as np
from typing import Any, Optional
from infrastructure.ml.classifiers.base import BaseClassifier
from infrastructure.ml.processors.data_processor import DashSpectrumProcessor
from infrastructure.ml.classifiers.architectures import AstroDashPyTorchNet
import logging
from config.settings import get_settings, Settings

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

    def _default_model_path(self):
        return self.config.dash_model_path

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Dash model not found at {self.model_path}. Classifier will not work.")
            self.model = None
            return
        state_dict = torch.load(self.model_path, map_location=self.device)
        n_types = state_dict['classifier.3.weight'].shape[0]
        self.model = AstroDashPyTorchNet(n_types)
        self.model.load_state_dict(state_dict)
        self.model.eval()
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
        # For now, just return the top 3 classes and probabilities
        top_indices = np.argsort(softmax)[::-1][:3]
        results = {
            "best_matches": [
                {"class_index": int(idx), "probability": float(softmax[idx])}
                for idx in top_indices
            ],
            "probabilities": softmax.tolist(),
        }
        return results
