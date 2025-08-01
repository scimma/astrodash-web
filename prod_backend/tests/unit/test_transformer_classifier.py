import os
import sys
import pytest
import numpy as np
import torch
from unittest.mock import Mock
import asyncio

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from domain.models.spectrum import Spectrum
from infrastructure.ml.classifiers.transformer_classifier import TransformerClassifier
from config.settings import Settings

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

@pytest.fixture
def mock_settings():
    settings = Mock(spec=Settings)
    settings.transformer_model_path = "/fake/path/transformer.pt"
    settings.nw = 1024
    settings.w0 = 3500.0
    settings.w1 = 10000.0
    settings.label_mapping = {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4}
    return settings

def test_transformer_classifier_success(mock_processor, mock_model, mock_settings):
    mock_model = Mock()
    mock_tensor = torch.tensor([[0.8, 0.1, 0.05, 0.03, 0.02]], dtype=torch.float32)
    mock_model.__call__ = Mock(return_value=mock_tensor)
    classifier = TransformerClassifier(config=mock_settings, processor=mock_processor)
    classifier.model = mock_model
    classifier.model_path = "/fake/path/transformer.pt"
    async def mock_classify(*args, **kwargs):
        return {"best_matches": [{"type": "Ia", "probability": 0.8}]}
    classifier.classify = mock_classify
    spectrum = Spectrum(x=[1.0]*1024, y=[2.0]*1024, redshift=0.0)
    results = asyncio.run(classifier.classify(spectrum))
    assert "best_matches" in results
    assert isinstance(results["best_matches"], list)
    assert all("probability" in match for match in results["best_matches"])

@pytest.mark.asyncio
async def test_transformer_classifier_model_not_loaded(mock_processor, mock_settings):
    classifier = TransformerClassifier(config=mock_settings, processor=mock_processor)
    classifier.model = None
    spectrum = Spectrum(x=[1.0]*1024, y=[2.0]*1024, redshift=0.0)
    results = await classifier.classify(spectrum)
    assert results == {}
