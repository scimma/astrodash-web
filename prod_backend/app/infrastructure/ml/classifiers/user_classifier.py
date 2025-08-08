import os
import json
import torch
import numpy as np
from typing import Any, Optional
from app.infrastructure.ml.classifiers.base import BaseClassifier
from app.config.settings import get_settings, Settings
from app.config.logging import get_logger
from app.core.exceptions import FileNotFoundException

logger = get_logger(__name__)

class UserClassifier(BaseClassifier):
    def __init__(self, user_model_id: str, config: Settings = None):
        super().__init__(config)
        self.user_model_id = user_model_id
        self.config = config or get_settings()
        self.model_dir = self.config.user_model_dir
        # Create subdirectory structure to match ModelStorage
        self.model_base = os.path.join(self.model_dir, self.user_model_id, self.user_model_id)
        self.model_path = self.model_base + '.pth'
        self.mapping_path = self.model_base + '.classes.json'
        self.input_shape_path = self.model_base + '.input_shape.json'
        self.model = None
        self.class_map = None
        self.input_shape = None
        self._load_model_and_metadata()

    def _load_model_and_metadata(self):
        if not os.path.exists(self.model_path):
            logger.error(f"User model file not found: {self.model_path}")
            raise FileNotFoundException(self.model_path)
        if not os.path.exists(self.mapping_path):
            logger.error(f"User model class mapping not found: {self.mapping_path}")
            raise FileNotFoundException(self.mapping_path)
        if not os.path.exists(self.input_shape_path):
            logger.error(f"User model input shape not found: {self.input_shape_path}")
            raise FileNotFoundException(self.input_shape_path)
        self.model = torch.jit.load(self.model_path, map_location='cpu')
        self.model.eval()
        with open(self.mapping_path, 'r') as f:
            self.class_map = json.load(f)
        with open(self.input_shape_path, 'r') as f:
            self.input_shape = json.load(f)
        logger.info(f"Loaded user model {self.user_model_id} with input shape {self.input_shape}")

    async def classify(self, spectrum: Any) -> dict:
        try:
            flux = np.array(spectrum.y)
            wavelength = np.array(spectrum.x)
            redshift = getattr(spectrum, 'redshift', 0.0) or 0.0
            # Handle different input shapes based on model type
            if len(self.input_shape) == 4:  # [batch, channels, height, width] - CNN style
                flat_size = np.prod(self.input_shape[1:])
                flux_flat = np.zeros(flat_size)
                n = min(len(flux), flat_size)
                flux_flat[:n] = flux[:n]
                model_input = torch.tensor(flux_flat, dtype=torch.float32).reshape(self.input_shape)
                with torch.no_grad():
                    output = self.model(model_input)
                probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
            else:  # Transformer style - needs wavelength, flux, redshift
                # Interpolate to 1024 points if needed
                if len(flux) != 1024:
                    x_old = np.linspace(0, 1, len(flux))
                    x_new = np.linspace(0, 1, 1024)
                    flux = np.interp(x_new, x_old, flux)
                    wavelength = np.interp(x_new, x_old, wavelength)
                wavelength_tensor = torch.tensor(wavelength, dtype=torch.float32).unsqueeze(0)  # [1, 1024]
                flux_tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)              # [1, 1024]
                redshift_tensor = torch.tensor([[redshift]], dtype=torch.float32)               # [1, 1]
                with torch.no_grad():
                    output = self.model(wavelength_tensor, flux_tensor, redshift_tensor)
                probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
            idx_to_label = {v: k for k, v in self.class_map.items()}
            top_indices = np.argsort(probs)[::-1][:3]
            matches = []
            for idx in top_indices:
                class_name = idx_to_label.get(idx, f'unknown_class_{idx}')
                matches.append({
                    'type': class_name,
                    'age': 'N/A',  # User models don't classify age
                    'probability': float(probs[idx]),
                    'redshift': redshift,
                    'rlap': None,  # Not calculated for user-uploaded models
                    'reliable': bool(probs[idx] > 0.5)
                })
            best_match = matches[0] if matches else {'type': 'Unknown', 'age': 'N/A', 'probability': 0.0}
            return {
                "best_matches": matches,
                "best_match": best_match,
                "reliable_matches": best_match.get('reliable', False) if best_match else False,
                "user_model_id": self.user_model_id
            }
        except Exception as e:
            logger.error(f"Error using user-uploaded model: {e}", exc_info=True)
            raise
