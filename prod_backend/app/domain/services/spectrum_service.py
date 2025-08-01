from typing import Any, Optional
from app.domain.models.spectrum import Spectrum
from app.domain.repositories.spectrum_repository import SpectrumRepository
from app.config.settings import Settings, get_settings
import logging

logger = logging.getLogger("spectrum_service")

class SpectrumService:
    def __init__(self, file_repo: SpectrumRepository, osc_repo: SpectrumRepository, settings: Optional[Settings] = None):
        """Service for spectrum operations. Injects repositories and settings."""
        self.file_repo = file_repo
        self.osc_repo = osc_repo
        self.settings = settings or get_settings()

    async def get_spectrum_from_file(self, file: Any) -> Spectrum:
        logger.info(f"Getting spectrum from file: {getattr(file, 'filename', 'unknown')}")
        spectrum = await self.file_repo.get_from_file(file)
        logger.info(f"Repository returned spectrum: {spectrum}")

        if not spectrum:
            logger.error("Repository returned None spectrum")
            raise ValueError("Invalid or unreadable spectrum file.")

        logger.info(f"Spectrum is_valid() result: {spectrum.is_valid()}")
        if not spectrum.is_valid():
            logger.error("Spectrum validation failed")
            raise ValueError("Invalid or unreadable spectrum file.")

        logger.info("Spectrum validation passed")
        return spectrum

    async def get_spectrum_from_osc(self, osc_ref: str) -> Spectrum:
        logger.info(f"Spectrum service: Starting to get spectrum from OSC reference: {osc_ref}")
        spectrum = await self.osc_repo.get_by_osc_ref(osc_ref)
        logger.info(f"Spectrum service: OSC repository returned spectrum: {spectrum}")

        if not spectrum:
            logger.error(f"Spectrum service: OSC repository returned None spectrum for {osc_ref}")
            raise ValueError(f"Could not retrieve valid spectrum data from OSC for reference: {osc_ref}. The spectrum may not exist or the OSC API may be unavailable.")

        logger.info(f"Spectrum service: Spectrum validation result: {spectrum.is_valid()}")
        if not spectrum.is_valid():
            logger.error(f"Spectrum service: Spectrum validation failed for {osc_ref}")
            raise ValueError(f"Could not retrieve valid spectrum data from OSC for reference: {osc_ref}. The spectrum may not exist or the OSC API may be unavailable.")

        logger.info(f"Spectrum service: Successfully retrieved and validated OSC spectrum for {osc_ref}")
        return spectrum

    async def validate_spectrum(self, spectrum: Spectrum) -> bool:
        """
        Validate a spectrum's data (basic check).
        """
        return spectrum.is_valid()
