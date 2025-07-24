from abc import ABC, abstractmethod
from typing import Optional, Any
from domain.models.spectrum import Spectrum

class SpectrumRepository(ABC):
    @abstractmethod
    async def save(self, spectrum: Spectrum) -> Spectrum:
        """
        Save a spectrum to persistent storage.
        Not implemented in current backend (placeholder for future DB/file storage).
        """
        pass

    @abstractmethod
    async def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        """
        Retrieve a spectrum by its unique ID.
        Not implemented in current backend (placeholder for future DB/file storage).
        """
        pass

    @abstractmethod
    async def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        """
        Retrieve a spectrum from the OSC API by its reference string.
        """
        pass

    @abstractmethod
    async def get_from_file(self, file: Any) -> Optional[Spectrum]:
        """
        Retrieve a spectrum from an uploaded file (FITS, .dat, .txt, .lnw).
        """
        pass
