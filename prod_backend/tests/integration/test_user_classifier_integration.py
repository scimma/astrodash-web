import os
import sys
import pytest
import torch
import numpy as np
from unittest.mock import Mock
from unittest.mock import patch

# Set PYTHONPATH to the app directory if not already set
APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from domain.models.spectrum import Spectrum
from infrastructure.ml.classifiers.user_classifier import UserClassifier

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), '../files')
DAT_FILE = os.path.join(TEST_FILES_DIR, 'ptf10hgi.p67.dat')

@patch('app.infrastructure.ml.classifiers.user_classifier.torch.softmax', lambda x, dim: Mock(cpu=lambda: Mock(numpy=lambda: np.array([[0.8, 0.2]]))))
@pytest.mark.asyncio
async def test_user_classifier_integration():
    config = Mock()
    config.user_model_dir = "/home/jesusca/code_personal/astrodash-web/backend/astrodash_models/user_uploaded"

    # Mock the model loading to avoid file system dependencies
    with patch.object(UserClassifier, "_load_model_and_metadata", lambda self: None):
        classifier = UserClassifier(user_model_id="test_model", config=config)

        mock_classifier = Mock()
        mock_tensor = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
        mock_classifier.__call__ = Mock(return_value=mock_tensor)
        mock_classifier.classify = Mock(return_value={
            "best_matches": [{"type": "Ia", "probability": 0.8}],
            "probabilities": [0.8, 0.2]
        })

        classifier.model = mock_classifier
        classifier.class_map = {"Ia": 0, "II": 1}
        classifier.input_shape = [1, 1024]

        with open(DAT_FILE, 'r') as f:
            data = np.loadtxt(f)
            x = data[:, 0]
            y = data[:, 1]
        spectrum = Spectrum(x=x.tolist(), y=y.tolist(), redshift=0.0)
        results = await classifier.classify(spectrum)
        assert results is not None
        assert "best_matches" in results
        assert "best_match" in results
        assert "reliable_matches" in results
        assert "user_model_id" in results
        assert isinstance(results["best_matches"], list)
        assert len(results["best_matches"]) > 0
        assert all("probability" in match for match in results["best_matches"])
        assert all("type" in match for match in results["best_matches"])
        assert results["user_model_id"] == "test_model"
