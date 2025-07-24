import os
import torch
import numpy as np
from typing import Any, Optional
from infrastructure.ml.classifiers.base import BaseClassifier
from infrastructure.ml.processors.data_processor import TransformerSpectrumProcessor
from infrastructure.ml.classifiers.architectures import spectraTransformerEncoder
import logging
from config.settings import get_settings, Settings

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
        self.model = None
        self.model_path = self.config.transformer_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self.label_mapping = getattr(self.config, "label_mapping", {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4})
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}

    def _default_model_path(self):
        return self.config.transformer_model_path

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Transformer model not found at {self.model_path}. Classifier will not work.")
            self.model = None
            return
        # Model hyperparameters (should match training)
        bottleneck_length = self.config.get("bottleneck_length", 1)
        model_dim = self.config.get("model_dim", 128)
        num_heads = self.config.get("num_heads", 4)
        num_layers = self.config.get("num_layers", 6)
        num_classes = self.config.get("num_classes", 5)
        ff_dim = self.config.get("ff_dim", 256)
        dropout = self.config.get("dropout", 0.1)
        selfattn = self.config.get("selfattn", False)
        self.model = spectraTransformerEncoder(
            bottleneck_length=bottleneck_length,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            ff_dim=ff_dim,
            dropout=dropout,
            selfattn=selfattn
        ).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Transformer model loaded from {self.model_path}")

    async def classify(self, spectrum: Any) -> dict:
        if self.model is None:
            logger.error("Transformer model is not loaded. Returning empty result.")
            return {}
        x = np.array(spectrum.x)
        y = np.array(spectrum.y)
        z = getattr(spectrum, 'redshift', 0.0) or 0.0
        x_proc, y_proc, redshift = self.processor.process(x, y, z)
        wavelength = torch.tensor(x_proc, dtype=torch.float32).unsqueeze(0).to(self.device)
        flux = torch.tensor(y_proc, dtype=torch.float32).unsqueeze(0).to(self.device)
        redshift_tensor = torch.tensor([[redshift]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(wavelength, flux, redshift_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        top_indices = np.argsort(probs)[::-1][:3]
        results = {
            "best_matches": [
                {"type": self.idx_to_label.get(idx, f"unknown_class_{idx}"), "probability": float(probs[idx])}
                for idx in top_indices
            ],
            "probabilities": probs.tolist(),
        }
        return results
