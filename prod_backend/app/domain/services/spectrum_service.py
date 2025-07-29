from typing import Any, Optional
from app.domain.models.spectrum import Spectrum
from app.domain.repositories.spectrum_repository import SpectrumRepository
from app.config.settings import Settings, get_settings

class SpectrumService:
    def __init__(self, file_repo: SpectrumRepository, osc_repo: SpectrumRepository, settings: Optional[Settings] = None):
        """Service for spectrum operations. Injects repositories and settings."""
        self.file_repo = file_repo
        self.osc_repo = osc_repo
        self.settings = settings or get_settings()

    async def get_spectrum_from_file(self, file: Any) -> Spectrum:
        spectrum = await self.file_repo.get_from_file(file)
        if not spectrum or not spectrum.is_valid():
            raise ValueError("Invalid or unreadable spectrum file.")
        return spectrum

    async def get_spectrum_from_osc(self, osc_ref: str) -> Spectrum:
        spectrum = await self.osc_repo.get_by_osc_ref(osc_ref)
        if not spectrum or not spectrum.is_valid():
            raise ValueError(f"Invalid or unreadable OSC spectrum for ref: {osc_ref}")
        return spectrum

    async def validate_spectrum(self, spectrum: Spectrum) -> bool:
        """
        Validate a spectrum's data (basic check).
        """
        return spectrum.is_valid()
