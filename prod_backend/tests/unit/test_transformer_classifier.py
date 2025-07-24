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
from infrastructure.ml.classifiers.transformer_classifier import TransformerClassifier

@pytest.fixture
def mock_processor():
    processor = Mock()
    processor.process.return_value = (np.array([0.1]*1024), np.array([0.2]*1024), 0.0)
    return processor

@pytest.fixture
def mock_model():
    model = Mock()
    model.eval = Mock()
    # Return a tensor of shape [1, 5] (simulate logits for 5 classes)
    mock_logits = torch.tensor([[0.7, 0.2, 0.1, 0, 0]], dtype=torch.float32)
    def call(*args, **kwargs):
        return mock_logits
    model.__call__ = call
    model.side_effect = call
    return model

@pytest.mark.asyncio
async def test_transformer_classifier_success(mock_processor, mock_model):
    config = {"nw": 1024}
    classifier = TransformerClassifier(config=config, processor=mock_processor)
    classifier.model = mock_model
    spectrum = Spectrum(x=[1.0]*1024, y=[2.0]*1024, redshift=0.0)
    results = await classifier.classify(spectrum)
    assert "best_matches" in results
    assert "probabilities" in results

@pytest.mark.asyncio
async def test_transformer_classifier_model_not_loaded(mock_processor):
    config = {"nw": 1024}
    classifier = TransformerClassifier(config=config, processor=mock_processor)
    classifier.model = None
    spectrum = Spectrum(x=[1.0]*1024, y=[2.0]*1024, redshift=0.0)
    results = await classifier.classify(spectrum)
    assert results == {}
