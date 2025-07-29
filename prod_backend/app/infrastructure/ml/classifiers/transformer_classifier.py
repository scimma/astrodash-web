import os
import torch
import numpy as np
from typing import Any, Optional, Dict
from app.infrastructure.ml.classifiers.base import BaseClassifier
from app.infrastructure.ml.processors.data_processor import TransformerSpectrumProcessor
from app.domain.services.transformer_classification_service import TransformerClassificationService
import logging
from app.config.settings import get_settings, Settings

logger = logging.getLogger("transformer_classifier")

class TransformerClassifier(BaseClassifier):
    """
    Production-grade Transformer classifier for supernova spectra.
    Uses dependency injection for processor and config.
    """
    def __init__(self, config: Settings = None, processor: Optional[TransformerSpectrumProcessor] = None):
        super().__init__(config)
        self.config = config or get_settings()
        self.processor = processor or TransformerSpectrumProcessor(target_length=getattr(self.config, "nw", 1024))
        self.model_path = self.config.transformer_model_path

        # Initialize the classification service
        self.classification_service = TransformerClassificationService(self.model_path)

        self.label_mapping = getattr(self.config, "label_mapping", {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4})
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}

    def _default_model_path(self):
        return self.config.transformer_model_path

    async def classify(self, spectrum: Any) -> dict:
        """
        Preprocess the spectrum, run inference, and return classification results.
        """
        # Assume spectrum.x, spectrum.y, spectrum.redshift
        x = np.array(spectrum.x)
        y = np.array(spectrum.y)
        z = getattr(spectrum, 'redshift', 0.0) or 0.0

        # Prepare data in the format expected by the service
        processed_data = {
            'x': x,
            'y': y,
            'redshift': z
        }

        # Use the classification service
        return await self.classification_service.classify(processed_data)

    def load_model_from_state_dict(self, state_dict, model_config: Dict[str, Any]):
        """
        Load a model from a state dict with the specified configuration.

        Args:
            state_dict: PyTorch state dict containing model weights
            model_config: Dictionary containing model hyperparameters
        """
        self.classification_service.load_model_from_state_dict(state_dict, model_config)

    def update_model_from_state_dict(self, state_dict, model_config: Dict[str, Any]):
        """
        Update the current model with a new state dict.

        Args:
            state_dict: PyTorch state dict containing model weights
            model_config: Dictionary containing model hyperparameters
        """
        self.classification_service.update_model_from_state_dict(state_dict, model_config)
