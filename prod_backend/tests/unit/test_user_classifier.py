import os
import sys
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from domain.models.spectrum import Spectrum
from infrastructure.ml.classifiers.user_classifier import UserClassifier

USER_MODEL_ID = "932eed3d-4d0e-4594-a490-5fd4f5e7a344"

@pytest.fixture
def mock_cnn_user_classifier():
    with patch.object(UserClassifier, "_load_model_and_metadata", lambda self: None):
        classifier = UserClassifier(user_model_id="dummy", config={})
        classifier.model = Mock()
        classifier.class_map = {"Ia": 0, "II": 1}
        classifier.input_shape = [1, 1, 32, 32]
        def call(input_tensor):
            return torch.tensor([[0.8, 0.2]], dtype=torch.float32)
        classifier.model.__call__ = call
        classifier.model.side_effect = call
        return classifier

@pytest.fixture
def mock_transformer_user_classifier():
    with patch.object(UserClassifier, "_load_model_and_metadata", lambda self: None):
        classifier = UserClassifier(user_model_id="dummy", config={})
        classifier.model = Mock()
        classifier.class_map = {"Ia": 0, "II": 1}
        classifier.input_shape = [1, 1024]
        def call(wavelength, flux, redshift):
            return torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        classifier.model.__call__ = call
        classifier.model.side_effect = call
        return classifier

@pytest.mark.asyncio
async def test_user_classifier_cnn(mock_cnn_user_classifier):
    spectrum = Spectrum(x=[1.0]*1024, y=[2.0]*1024, redshift=0.0)
    results = await mock_cnn_user_classifier.classify(spectrum)
    assert "best_matches" in results
    assert results["best_matches"][0]["type"] in ("Ia", "II")
    assert "probabilities" not in results  # UserClassifier does not return full probabilities

@pytest.mark.asyncio
async def test_user_classifier_transformer(mock_transformer_user_classifier):
    spectrum = Spectrum(x=[1.0]*1024, y=[2.0]*1024, redshift=0.0)
    results = await mock_transformer_user_classifier.classify(spectrum)
    assert "best_matches" in results
    assert results["best_matches"][0]["type"] in ("Ia", "II")
    assert "probabilities" not in results

@pytest.mark.asyncio
async def test_user_classifier_error():
    with patch.object(UserClassifier, "_load_model_and_metadata", lambda self: None):
        classifier = UserClassifier(user_model_id="dummy", config={})
        classifier.model = None
        classifier.class_map = None
        classifier.input_shape = None
        spectrum = Spectrum(x=[1.0]*1024, y=[2.0]*1024, redshift=0.0)
        with pytest.raises(Exception):
            await classifier.classify(spectrum)
