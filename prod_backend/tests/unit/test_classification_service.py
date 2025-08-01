import os
import sys
import pytest
from unittest.mock import AsyncMock, Mock
# Set PYTHONPATH to the app directory if not already set
APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from domain.models.spectrum import Spectrum
from domain.models.classification import Classification
from domain.services.classification_service import ClassificationService

@pytest.fixture
def mock_model_factory():
    return Mock()

@pytest.fixture
def mock_classifier():
    classifier = Mock()
    classifier.classify = AsyncMock(return_value={"best_matches": [{"type": "Ia", "probability": 0.9}], "probabilities": {"Ia": 0.9, "II": 0.1}})
    return classifier

@pytest.fixture
def spectrum():
    return Spectrum(x=[1.0, 2.0], y=[3.0, 4.0], redshift=0.1, id="spec-1")

@pytest.mark.asyncio
async def test_classify_spectrum_success(mock_model_factory, mock_classifier, spectrum):
    mock_model_factory.get_classifier.return_value = mock_classifier
    service = ClassificationService(mock_model_factory)
    result = await service.classify_spectrum(spectrum, model_type="dash")
    assert result is not None
    assert result.spectrum_id == "spec-1"
    assert result.model_type == "dash"
    assert "best_matches" in result.results
    assert "probabilities" in result.results
    assert isinstance(result.results["best_matches"], list)
    assert len(result.results["best_matches"]) > 0
    assert "type" in result.results["best_matches"][0]
    assert "probability" in result.results["best_matches"][0]
    mock_model_factory.get_classifier.assert_called_once_with("dash", None)
    mock_classifier.classify.assert_awaited_once_with(spectrum)

@pytest.mark.asyncio
async def test_classify_spectrum_with_user_model(mock_model_factory, mock_classifier, spectrum):
    mock_model_factory.get_classifier.return_value = mock_classifier
    service = ClassificationService(mock_model_factory)
    user_model_id = "user-123"
    result = await service.classify_spectrum(spectrum, model_type="transformer", user_model_id=user_model_id)
    assert result.user_model_id == user_model_id
    mock_model_factory.get_classifier.assert_called_once_with("transformer", user_model_id)
    mock_classifier.classify.assert_awaited_once_with(spectrum)

@pytest.mark.asyncio
async def test_classify_spectrum_failure(mock_model_factory, mock_classifier, spectrum):
    mock_classifier.classify = AsyncMock(return_value={})  # Return empty dict instead of None
    mock_model_factory.get_classifier.return_value = mock_classifier
    service = ClassificationService(mock_model_factory)
    with pytest.raises(ValueError):
        await service.classify_spectrum(spectrum, model_type="dash")
