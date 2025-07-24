import os
import sys
import pytest
# Set PYTHONPATH to the app directory if not already set
APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from domain.services.spectrum_service import SpectrumService
from infrastructure.storage.file_spectrum_repository import FileSpectrumRepository, OSCSpectrumRepository

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
