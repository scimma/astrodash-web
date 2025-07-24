import os
import sys
import pytest
import numpy as np
import torch
from unittest.mock import Mock

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from domain.models.spectrum import Spectrum
from infrastructure.ml.classifiers.dash_classifier import DashClassifier

@pytest.fixture
def mock_processor():
    processor = Mock()
    processor.process.return_value = (np.array([0.1]*1024), 0, 1023, 0.0)
    return processor

@pytest.fixture
def mock_model():
    model = Mock()
    model.eval = Mock()
    # Return a tensor of shape [1, 1024] (simulate model output)
    mock_tensor = torch.tensor([[0.8] + [0.1] * 2 + [0] * 1021], dtype=torch.float32)
    def call(input_tensor):
        return mock_tensor
    model.__call__ = call
    model.side_effect = call
    return model

@pytest.mark.asyncio
async def test_dash_classifier_success(mock_processor, mock_model):
    config = {"nw": 1024, "w0": 3500.0, "w1": 10000.0}
    classifier = DashClassifier(config=config, processor=mock_processor)
    classifier.model = mock_model
    spectrum = Spectrum(x=[1.0]*1024, y=[2.0]*1024, redshift=0.0)
    results = await classifier.classify(spectrum)
    assert "best_matches" in results
    assert "probabilities" in results

@pytest.mark.asyncio
async def test_dash_classifier_model_not_loaded(mock_processor):
    config = {"nw": 1024, "w0": 3500.0, "w1": 10000.0}
    classifier = DashClassifier(config=config, processor=mock_processor)
    classifier.model = None
    spectrum = Spectrum(x=[1.0]*1024, y=[2.0]*1024, redshift=0.0)
    results = await classifier.classify(spectrum)
    assert results == {}
