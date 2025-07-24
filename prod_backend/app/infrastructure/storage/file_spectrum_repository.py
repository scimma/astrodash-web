from typing import Optional, Any
from domain.models.spectrum import Spectrum
from domain.repositories.spectrum_repository import SpectrumRepository
from infrastructure.ml.processors.spectrum_processor import SpectrumProcessor

class FileSpectrumRepository(SpectrumRepository):
    def __init__(self):
        self.processor = SpectrumProcessor()

    async def save(self, spectrum: Spectrum) -> Spectrum:
        # Not implemented: placeholder for file/DB storage
        raise NotImplementedError("Saving spectra is not implemented.")

    async def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        # Not implemented: placeholder for file/DB storage
        raise NotImplementedError("Retrieving by ID is not implemented.")

    async def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        # Not implemented in file repository
        return None

    async def get_from_file(self, file: Any) -> Optional[Spectrum]:
        data = self.processor.read_file(file)
        if data:
            return Spectrum(
                x=data.get('x'),
                y=data.get('y'),
                redshift=data.get('redshift', None)
            )
        return None

class OSCSpectrumRepository(SpectrumRepository):
    def __init__(self):
        self.processor = SpectrumProcessor()

    async def save(self, spectrum: Spectrum) -> Spectrum:
        # Not implemented: placeholder for OSC
        raise NotImplementedError("Saving spectra is not implemented.")

    async def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        # Not implemented: placeholder for OSC
        raise NotImplementedError("Retrieving by ID is not implemented.")

    async def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        data = self.processor.read_file(osc_ref)
        if data:
            return Spectrum(
                x=data.get('x'),
                y=data.get('y'),
                redshift=data.get('redshift', None)
            )
        return None

    async def get_from_file(self, file: Any) -> Optional[Spectrum]:
        # Not implemented in OSC repository
        return None
