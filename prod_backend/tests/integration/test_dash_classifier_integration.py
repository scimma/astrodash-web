import os
import sys
import pytest
import numpy as np
from unittest.mock import Mock

# Set PYTHONPATH to the app directory if not already set
APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from domain.models.spectrum import Spectrum
from infrastructure.ml.classifiers.dash_classifier import DashClassifier
from infrastructure.ml.processors.data_processor import DashSpectrumProcessor
from config.settings import Settings

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), '../files')
DAT_FILE = os.path.join(TEST_FILES_DIR, 'ptf10hgi.p67.dat')

@pytest.mark.asyncio
async def test_dash_classifier_with_real_file():
    config = Mock(spec=Settings)
    config.dash_model_path = "/home/jesusca/code_personal/astrodash-web/backend/astrodash_models/zeroZ/pytorch_model.pth"
    config.nw = 1024
    config.w0 = 3500.0
    config.w1 = 10000.0
    
    processor = DashSpectrumProcessor(w0=config.w0, w1=config.w1, nw=config.nw)
    classifier = DashClassifier(config=config, processor=processor)
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
    assert "probabilities" in results
    assert isinstance(results["probabilities"], list)
