import os
import sys
import pytest
from unittest.mock import Mock, AsyncMock
# Set PYTHONPATH to the app directory if not already set
APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from domain.services.spectrum_service import SpectrumService
from infrastructure.storage.file_spectrum_repository import FileSpectrumRepository, OSCSpectrumRepository
from domain.models.spectrum import Spectrum
from shared.utils.validators import validate_spectrum

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), '../files')
DAT_FILE = os.path.join(TEST_FILES_DIR, 'ptf10hgi.p67.dat')
LNW_FILE = os.path.join(TEST_FILES_DIR, 'sn00cp_bsnip.lnw')
OSC_REF = 'osc-sn2002er-0'

@pytest.mark.asyncio
async def test_get_spectrum_from_file_dat():
    file_repo = FileSpectrumRepository()
    osc_repo = OSCSpectrumRepository()
    service = SpectrumService(file_repo, osc_repo)
    with open(DAT_FILE, 'rb') as f:
        spectrum = await service.get_spectrum_from_file(f)
    assert spectrum is not None
    validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)
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
    validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)
    assert len(spectrum.x) > 0
    assert len(spectrum.y) > 0

@pytest.mark.asyncio
async def test_get_spectrum_from_osc():
    # Mock the OSC repository to avoid network dependencies
    mock_osc_repo = Mock(spec=OSCSpectrumRepository)
    mock_osc_repo.get_by_osc_ref = AsyncMock(return_value=Spectrum(
        x=[4000.0, 5000.0, 6000.0],
        y=[1.0, 2.0, 3.0],
        redshift=0.1,
        osc_ref=OSC_REF
    ))

    file_repo = FileSpectrumRepository()
    service = SpectrumService(file_repo, mock_osc_repo)
    spectrum = await service.get_spectrum_from_osc(OSC_REF)
    assert spectrum is not None
    validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)
    assert len(spectrum.x) > 0
    assert len(spectrum.y) > 0
