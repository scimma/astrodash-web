from typing import Any, List
from pydantic import validator, ValidationError
import numpy as np
import torch
import os


def validate_spectrum_data(x: List[float], y: List[float]) -> None:
    """Raise ValidationError if spectrum data is invalid."""
    if not x or not y or len(x) != len(y):
        raise ValidationError("Spectrum x and y must be non-empty and of equal length.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValidationError("Spectrum data contains NaN values.")


def validate_redshift(redshift: Any) -> float:
    """Validate and return a proper redshift value (>= 0)."""
    try:
        z = float(redshift)
        if z < 0:
            raise ValidationError("Redshift must be non-negative.")
        return z
    except Exception:
        raise ValidationError("Invalid redshift value.")


def validate_file_extension(filename: str, allowed: List[str] = [".dat", ".lnw", ".txt"]) -> None:
    """Raise ValidationError if file extension is not allowed."""
    if not any(filename.lower().endswith(ext) for ext in allowed):
        raise ValidationError(f"File extension not allowed. Allowed: {allowed}")


def non_empty_list(cls, v):
    """Pydantic validator: ensure a list is not empty."""
    if not v or not isinstance(v, list) or len(v) == 0:
        raise ValueError('List must not be empty')
    return v


def validate_user_model(model_path: str, input_shape: List[int], allowed_exts: List[str] = [".pth", ".pt"]) -> None:
    """
    Validate a user-uploaded model by checking file extension, loading with torch.jit, and running a dummy input.
    Raises ValidationError if any check fails.
    """
    validate_file_extension(model_path, allowed_exts)
    if not os.path.exists(model_path):
        raise ValidationError(f"Model file does not exist: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        dummy_input = torch.randn(*input_shape)
        with torch.no_grad():
            output = model(dummy_input)
        if not hasattr(output, 'shape') or output.shape[0] != 1:
            raise ValidationError(f"Model output shape {getattr(output, 'shape', None)} is invalid.")
    except Exception as e:
        raise ValidationError(f"Failed to load or validate user model: {e}")
