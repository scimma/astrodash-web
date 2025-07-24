import os
import sys
import pytest
import numpy as np

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
    config = {
        "transformer_model_path": "/home/jesusca/code_personal/astrodash-web/backend/astrodash_models/yuqing_models/TF_wiserep_v6.pt",
        "nw": 1024,
        "label_mapping": {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4},
        "bottleneck_length": 1,
        "model_dim": 128,
        "num_heads": 4,
        "num_layers": 6,
        "num_classes": 5,
        "ff_dim": 256,
        "dropout": 0.1,
        "selfattn": False,
    }
    processor = TransformerSpectrumProcessor(target_length=config["nw"])
    classifier = TransformerClassifier(config=config, processor=processor)
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
