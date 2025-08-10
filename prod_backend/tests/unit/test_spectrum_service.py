import pytest
import os
import sys
from unittest.mock import AsyncMock, Mock

# Set PYTHONPATH to the app directory if not already set
APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

# Import the modules after setting PYTHONPATH
from domain.models.spectrum import Spectrum
from domain.services.spectrum_service import SpectrumService
from infrastructure.storage.file_spectrum_repository import FileSpectrumRepository, OSCSpectrumRepository

OSC_REF = 'osc-sn2002er-0'

# --- Unit tests with mocks ---
@pytest.fixture
def valid_spectrum():
    s = Mock(spec=Spectrum)
    s.is_valid.return_value = True
    return s

@pytest.fixture
def invalid_spectrum():
    s = Mock(spec=Spectrum)
    s.is_valid.return_value = False
    return s

@pytest.fixture
def file_repo(valid_spectrum, invalid_spectrum):
    repo = Mock()
    repo.get_from_file = AsyncMock(return_value=valid_spectrum)
    return repo

@pytest.fixture
def osc_repo(valid_spectrum, invalid_spectrum):
    repo = Mock()
    repo.get_by_osc_ref = AsyncMock(return_value=valid_spectrum)
    return repo

@pytest.mark.asyncio
async def test_get_spectrum_from_file_success(file_repo, osc_repo, valid_spectrum):
    service = SpectrumService(file_repo, osc_repo)
    result = await service.get_spectrum_from_file(file=Mock())
    assert result is valid_spectrum
    file_repo.get_from_file.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_spectrum_from_osc_success(file_repo, osc_repo, valid_spectrum):
    service = SpectrumService(file_repo, osc_repo)
    result = await service.get_spectrum_from_osc(osc_ref=OSC_REF)
    assert result is valid_spectrum
    osc_repo.get_by_osc_ref.assert_awaited_once_with(OSC_REF)

@pytest.mark.asyncio
async def test_get_spectrum_from_file_invalid(file_repo, osc_repo, invalid_spectrum):
    file_repo.get_from_file = AsyncMock(return_value=invalid_spectrum)
    service = SpectrumService(file_repo, osc_repo)
    with pytest.raises(ValueError):
        await service.get_spectrum_from_file(file=Mock())

@pytest.mark.asyncio
async def test_get_spectrum_from_osc_invalid(file_repo, osc_repo, invalid_spectrum):
    osc_repo.get_by_osc_ref = AsyncMock(return_value=invalid_spectrum)
    service = SpectrumService(file_repo, osc_repo)
    with pytest.raises(ValueError):
        await service.get_spectrum_from_osc(osc_ref=OSC_REF)

@pytest.mark.asyncio
async def test_validate_spectrum(valid_spectrum, invalid_spectrum):
    """Test spectrum validation using centralized validator function."""
    from app.shared.utils.validators import validate_spectrum

    # Valid spectrum should not raise exception
    try:
        validate_spectrum(valid_spectrum.x, valid_spectrum.y, valid_spectrum.redshift)
        validation_passed = True
    except Exception:
        validation_passed = False
    assert validation_passed is True

    # Invalid spectrum should raise exception
    try:
        validate_spectrum(invalid_spectrum.x, invalid_spectrum.y, invalid_spectrum.redshift)
        validation_passed = True
    except Exception:
        validation_passed = False
    assert validation_passed is False
