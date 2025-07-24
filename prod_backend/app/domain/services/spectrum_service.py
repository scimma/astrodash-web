from typing import Any, Optional
from domain.models.spectrum import Spectrum
from domain.repositories.spectrum_repository import SpectrumRepository

class SpectrumService:
    """
    Service layer for spectrum-related operations.
    Orchestrates spectrum retrieval and validation using the repository pattern.
    """
    def __init__(self, file_repo: SpectrumRepository, osc_repo: SpectrumRepository):
        self.file_repo = file_repo
        self.osc_repo = osc_repo

    async def get_spectrum_from_file(self, file: Any) -> Spectrum:
        """
        Retrieve and validate a spectrum from an uploaded file.
        Raises ValueError if the spectrum is invalid or cannot be read.
        """
        spectrum = await self.file_repo.get_from_file(file)
        if not spectrum or not spectrum.is_valid():
            raise ValueError("Invalid or unreadable spectrum file.")
        return spectrum

    async def get_spectrum_from_osc(self, osc_ref: str) -> Spectrum:
        """
        Retrieve and validate a spectrum from the OSC API by reference.
        Raises ValueError if the spectrum is invalid or cannot be fetched.
        """
        spectrum = await self.osc_repo.get_by_osc_ref(osc_ref)
        if not spectrum or not spectrum.is_valid():
            raise ValueError(f"Invalid or unreadable OSC spectrum for ref: {osc_ref}")
        return spectrum

    async def validate_spectrum(self, spectrum: Spectrum) -> bool:
        """
        Validate a spectrum's data (basic check).
        """
        return spectrum.is_valid()
