import os
import sys
import pytest
import numpy as np
from unittest.mock import Mock
import torch

# Set PYTHONPATH to the app directory if not already set
APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from domain.models.spectrum import Spectrum
from infrastructure.ml.classifiers.transformer_classifier import TransformerClassifier
from infrastructure.ml.processors.data_processor import TransformerSpectrumProcessor

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), '../files')
DAT_FILE = os.path.join(TEST_FILES_DIR, 'ptf10hgi.p67.dat')

@pytest.mark.asyncio
async def test_transformer_classifier_with_real_file():
    config = Mock()
    config.transformer_model_path = "/home/jesusca/code_personal/astrodash-web/backend/astrodash_models/yuqing_models/TF_wiserep_v6.pt"
    config.nw = 1024
    config.w0 = 3500.0
    config.w1 = 10000.0
    config.label_mapping = {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4}

    processor = TransformerSpectrumProcessor(target_length=config.nw)
    classifier = TransformerClassifier(config=config, processor=processor)

    # Mock the model to avoid file loading issues
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_tensor = torch.tensor([[0.8, 0.1, 0.05, 0.03, 0.02]], dtype=torch.float32)
    mock_model.__call__ = Mock(return_value=mock_tensor)
    classifier.model = mock_model

    with open(DAT_FILE, 'r') as f:
        data = np.loadtxt(f)
        x = data[:, 0]
        y = data[:, 1]
    spectrum = Spectrum(x=x.tolist(), y=y.tolist(), redshift=0.0)
    results = await classifier.classify(spectrum)
    assert results is not None
    assert "best_matches" in results
    assert len(results["best_matches"]) > 0
    assert all("probability" in match for match in results["best_matches"])
    assert "best_matches" in results
    # assert isinstance(results["probabilities"], list)  # Commented out - classifier returns best_matches, not probabilities
