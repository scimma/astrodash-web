import pytest
import os
from unittest.mock import AsyncMock, Mock
from domain.models.spectrum import Spectrum
from domain.services.spectrum_service import SpectrumService
from infrastructure.storage.file_spectrum_repository import FileSpectrumRepository, OSCSpectrumRepository

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), '../files')
DAT_FILE = os.path.join(TEST_FILES_DIR, 'ptf10hgi.p67.dat')
LNW_FILE = os.path.join(TEST_FILES_DIR, 'sn00cp_bsnip.lnw')
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
async def test_validate_spectrum(file_repo, osc_repo, valid_spectrum, invalid_spectrum):
    service = SpectrumService(file_repo, osc_repo)
    assert await service.validate_spectrum(valid_spectrum) is True
    assert await service.validate_spectrum(invalid_spectrum) is False

# --- Integration-style tests with real files and OSC ---
@pytest.mark.asyncio
async def test_get_spectrum_from_file_dat():
    file_repo = FileSpectrumRepository()
    osc_repo = OSCSpectrumRepository()
    service = SpectrumService(file_repo, osc_repo)
    with open(DAT_FILE, 'rb') as f:
        spectrum = await service.get_spectrum_from_file(f)
    assert spectrum is not None
    assert spectrum.is_valid()
    assert len(spectrum.x) > 0
    assert len(spectrum.y) > 0

@pytest.mark.asyncio
async def test_get_spectrum_from_file_lnw():
    file_repo = FileSpectrumRepository()
    osc_repo = OSCSpectrumRepository()
    service = SpectrumService(file_repo, osc_repo)
    with open(LNW_FILE, 'rb') as f:
        spectrum = await service.get_spectrum_from_file(f)
    assert spectrum is not None
    assert spectrum.is_valid()
    assert len(spectrum.x) > 0
    assert len(spectrum.y) > 0

@pytest.mark.asyncio
async def test_get_spectrum_from_osc():
    file_repo = FileSpectrumRepository()
    osc_repo = OSCSpectrumRepository()
    service = SpectrumService(file_repo, osc_repo)
    spectrum = await service.get_spectrum_from_osc(OSC_REF)
    assert spectrum is not None
    assert spectrum.is_valid()
    assert len(spectrum.x) > 0
    assert len(spectrum.y) > 0
