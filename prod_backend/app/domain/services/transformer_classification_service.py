import logging
from typing import Any, Dict, Optional
import numpy as np
import torch
from shared.utils.helpers import interpolate_to_1024
from infrastructure.ml.classifiers.architectures import spectraTransformerEncoder
import os

logger = logging.getLogger(__name__)


class TransformerClassificationService:
    """
    Service to orchestrate Transformer model classification, including preprocessing, inference, and result formatting.
    """
    
    def __init__(
        self,
        model_path: str,
        logger: Optional[logging.Logger] = None
    ):
        self.model_path = model_path
        self.logger = logger or logging.getLogger("transformer_classification_service")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()
        
        # Label mapping for 5 classes
        self.label_mapping = {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        self.logger.info(f"Using label mapping: {self.label_mapping}")

    def _load_model(self):
        """Load the transformer model with proper initialization"""
        if not self.model_path or not os.path.exists(self.model_path):
            self.logger.warning(f"Transformer model file not found at {self.model_path}. Model will not be available.")
            self.model = None
            return
            
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

            self.model = model
            self.logger.info(f"Transformer model loaded successfully from {self.model_path} with {sum(p.numel() for p in model.parameters())} parameters")

        except Exception as e:
            self.logger.error(f"Failed to load transformer model: {e}")
            self.model = None

    async def classify(
        self,
        processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify a spectrum using the Transformer model.
        
        Args:
            processed_data: Dictionary containing 'x' (wavelength), 'y' (flux), 'redshift'
            
        Returns:
            Dictionary with classification results
        """
        if self.model is None:
            self.logger.error("Transformer model is not loaded. Returning empty result.")
            return {}
            
        try:
            # Extract and preprocess data
            wavelength_data = interpolate_to_1024(processed_data['x'])  # wavelength
            flux_data = interpolate_to_1024(processed_data['y'])        # flux
            redshift = processed_data.get('redshift', 0.0)

            # Convert to tensors with proper shape
            wavelength = torch.tensor(wavelength_data, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 1024]
            flux = torch.tensor(flux_data, dtype=torch.float32).unsqueeze(0).to(self.device)              # [1, 1024]
            redshift_tensor = torch.tensor([[redshift]], dtype=torch.float32).to(self.device)             # [1, 1]

            self.logger.info(f"Input shapes - wavelength: {wavelength.shape}, flux: {flux.shape}, redshift: {redshift_tensor.shape}")
            self.logger.info(f"Redshift value: {redshift}")

            # Run inference
            with torch.no_grad():
                logits = self.model(wavelength, flux, redshift_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            self.logger.info(f"Model output probabilities: {probs}")

            # Get top 3 predictions using the correct label mapping
            top_indices = np.argsort(probs)[::-1][:3]
            matches = []

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

            self.logger.info(f"Classification results - best match: {best_match}")

            return {
                'best_matches': matches,
                'best_match': best_match,
                'reliable_matches': best_match.get('reliable', False) if best_match else False
            }
            
        except Exception as e:
            self.logger.error(f"Error during transformer classification: {e}")
            return {}

    def load_model_from_state_dict(self, state_dict, model_config: Dict[str, Any]):
        """
        Load a model from a state dict with the specified configuration.
        
        Args:
            state_dict: PyTorch state dict containing model weights
            model_config: Dictionary containing model hyperparameters
        """
        try:
            # Initialize the model with the provided config
            model = spectraTransformerEncoder(**model_config).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            self.model = model
            self.logger.info(f"Transformer model loaded from state dict with {sum(p.numel() for p in model.parameters())} parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to load transformer model from state dict: {e}")
            self.model = None

    def update_model_from_state_dict(self, state_dict, model_config: Dict[str, Any]):
        """
        Update the current model with a new state dict.
        
        Args:
            state_dict: PyTorch state dict containing model weights
            model_config: Dictionary containing model hyperparameters
        """
        self.load_model_from_state_dict(state_dict, model_config) 