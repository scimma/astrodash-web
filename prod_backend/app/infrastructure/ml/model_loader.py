import torch
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger("model_loader")

class ModelLoader:
    """
    Infrastructure component for loading and validating PyTorch models.
    Handles model loading, shape inference, and validation operations.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load a PyTorch model from app.file.
        
        Args:
            model_path: Path to the model file (.pth or .pt)
            
        Returns:
            Loaded PyTorch model
            
        Raises:
            ValueError: If model cannot be loaded
        """
        try:
            if not os.path.exists(model_path):
                raise ValueError(f"Model file does not exist: {model_path}")
            
            # Try loading as TorchScript first
            try:
                model = torch.jit.load(model_path, map_location=self.device)
                logger.info(f"Loaded TorchScript model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load as TorchScript: {e}")
                # Try loading as state dict
                state_dict = torch.load(model_path, map_location=self.device)
                # This would need model architecture info to reconstruct
                raise ValueError("State dict loading not implemented yet")
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def validate_model_with_inputs(
        self, 
        model: torch.nn.Module, 
        input_shapes: List[List[int]], 
        class_mapping: Dict[str, int]
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Validate model by running dummy inputs and checking output shape.
        
        Args:
            model: Loaded PyTorch model
            input_shapes: List of input shapes for each model input
            class_mapping: Dictionary mapping class names to indices
            
        Returns:
            Tuple of (output_shape, model_info)
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Prepare dummy inputs
            dummy_inputs = []
            for shape in input_shapes:
                dummy_inputs.append(torch.randn(*shape, device=self.device))
            
            # Run inference
            with torch.no_grad():
                if len(dummy_inputs) == 1:
                    output = model(dummy_inputs[0])
                elif len(dummy_inputs) == 2:
                    output = model(dummy_inputs[0], dummy_inputs[1])
                elif len(dummy_inputs) == 3:
                    output = model(dummy_inputs[0], dummy_inputs[1], dummy_inputs[2])
                else:
                    output = model(*dummy_inputs)
            
            # Extract output shape
            if hasattr(output, 'shape'):
                output_shape = list(output.shape)
            else:
                output_shape = [1]  # Fallback for scalar outputs
            
            # Validate output shape matches class mapping
            n_classes = len(class_mapping)
            if output_shape[-1] != n_classes:
                raise ValueError(
                    f"Model output shape {output_shape} does not match "
                    f"number of classes {n_classes} in class mapping."
                )
            
            # Collect model information
            model_info = {
                "input_shapes": input_shapes,
                "output_shape": output_shape,
                "n_classes": n_classes,
                "device": str(self.device),
                "model_type": "torchscript" if isinstance(model, torch.jit.ScriptModule) else "pytorch"
            }
            
            logger.info(f"Model validation successful. Output shape: {output_shape}")
            return output_shape, model_info
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise ValueError(f"Model validation failed: {str(e)}")
    
    def extract_model_metadata(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Extract metadata from a loaded model.
        
        Args:
            model: Loaded PyTorch model
            
        Returns:
            Dictionary containing model metadata
        """
        metadata = {
            "model_type": "torchscript" if isinstance(model, torch.jit.ScriptModule) else "pytorch",
            "device": str(self.device),
            "training": model.training,
        }
        
        # Try to get model parameters count
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            metadata.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            })
        except Exception as e:
            logger.warning(f"Could not extract parameter count: {e}")
        
        return metadata
    
    def cleanup_model(self, model: torch.nn.Module) -> None:
        """
        Clean up model resources.
        
        Args:
            model: Model to clean up
        """
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")

class ModelValidator:
    """
    Validator for model files and metadata.
    """
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str] = [".pth", ".pt"]) -> None:
        """
        Validate file extension.
        
        Args:
            filename: Name of the file to validate
            allowed_extensions: List of allowed extensions
            
        Raises:
            ValueError: If extension is not allowed
        """
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise ValueError(f"File extension {file_ext} not allowed. Allowed: {allowed_extensions}")
    
    @staticmethod
    def validate_class_mapping(class_mapping: Dict[str, int]) -> None:
        """
        Validate class mapping structure.
        
        Args:
            class_mapping: Dictionary mapping class names to indices
            
        Raises:
            ValueError: If class mapping is invalid
        """
        if not isinstance(class_mapping, dict) or not class_mapping:
            raise ValueError("Class mapping must be a non-empty dictionary")
        
        # Check for valid indices
        indices = list(class_mapping.values())
        if not all(isinstance(idx, int) and idx >= 0 for idx in indices):
            raise ValueError("All class mapping values must be non-negative integers")
        
        # Check for unique indices
        if len(indices) != len(set(indices)):
            raise ValueError("Class mapping indices must be unique")
        
        # Check for consecutive indices starting from 0
        expected_indices = set(range(len(indices)))
        if set(indices) != expected_indices:
            raise ValueError("Class mapping indices must be consecutive starting from 0")
    
    @staticmethod
    def validate_input_shape(input_shape: List[int]) -> None:
        """
        Validate input shape.
        
        Args:
            input_shape: List of integers representing input shape
            
        Raises:
            ValueError: If input shape is invalid
        """
        if not isinstance(input_shape, list) or not input_shape:
            raise ValueError("Input shape must be a non-empty list")
        
        if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
            raise ValueError("All input shape dimensions must be positive integers")