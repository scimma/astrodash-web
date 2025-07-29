from abc import ABC, abstractmethod
from typing import Optional, Any, List, Tuple, Dict
import numpy as np
import logging
from app.domain.models.spectrum import Spectrum
import os

logger = logging.getLogger(__name__)


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


class SpectrumTemplateInterface(ABC):
    """
    Abstract base class for spectrum template handling.
    Defines the contract for template loading and validation.
    """

    @abstractmethod
    async def get_template_spectrum(self, sn_type: str, age_bin: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get template spectrum for a specific SN type and age bin.

        Args:
            sn_type: Supernova type (e.g., 'Ia-norm', 'Ib-norm')
            age_bin: Age bin (e.g., '2 to 6', '-10 to -6')

        Returns:
            Tuple of (wavelength, flux) arrays
        """
        pass

    @abstractmethod
    async def get_all_templates(self) -> Dict[str, Any]:
        """
        Get all available templates.

        Returns:
            Dictionary containing all template data
        """
        pass

    @abstractmethod
    async def validate_template(self, sn_type: str, age_bin: str) -> bool:
        """
        Validate if a template exists for the given SN type and age bin.

        Args:
            sn_type: Supernova type
            age_bin: Age bin

        Returns:
            True if template exists and is valid
        """
        pass


class DASHSpectrumTemplate(SpectrumTemplateInterface):
    """
    DASH-specific template handler.
    Loads templates from the DASH model's template file.
    """

    def __init__(self, template_path: str):
        self.template_path = template_path
        self._templates = None
        logger.info(f"DASHSpectrumTemplate initialized with path: {template_path}")

    async def get_template_spectrum(self, sn_type: str, age_bin: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get template spectrum for DASH model."""
        try:
            templates = await self._load_templates()

            if sn_type not in templates:
                raise ValueError(f"SN type '{sn_type}' not found in templates")

            if age_bin not in templates[sn_type]:
                raise ValueError(f"Age bin '{age_bin}' not found for SN type '{sn_type}'")

            entry = templates[sn_type][age_bin]
            sn_info = entry.get('snInfo', None)

            if not isinstance(sn_info, np.ndarray) or sn_info.shape[0] == 0:
                raise ValueError(f"No template spectrum available for {sn_type} / {age_bin}")

            # Get the first template (placeholder for now)
            template = sn_info[0]
            wave = template[0]
            flux = template[1]

            logger.info(f"Template spectrum loaded for {sn_type} / {age_bin}")
            return wave, flux

        except Exception as e:
            logger.error(f"Error loading template spectrum: {e}")
            raise

    async def get_all_templates(self) -> Dict[str, Any]:
        """Get all DASH templates."""
        try:
            return await self._load_templates()
        except Exception as e:
            logger.error(f"Error loading all templates: {e}")
            raise

    async def validate_template(self, sn_type: str, age_bin: str) -> bool:
        """Validate DASH template."""
        try:
            templates = await self._load_templates()
            return (sn_type in templates and
                   age_bin in templates[sn_type] and
                   self._is_valid_entry(templates[sn_type][age_bin]))
        except Exception as e:
            logger.error(f"Error validating template: {e}")
            return False

    async def _load_templates(self) -> Dict[str, Any]:
        """Load templates from app.file."""
        if self._templates is None:
            try:
                data = np.load(self.template_path, allow_pickle=True)
                sn_templates_raw = data['snTemplates'].item()
                self._templates = {str(k): v for k, v in sn_templates_raw.items()}
                logger.info(f"Templates loaded: {list(self._templates.keys())}")
            except Exception as e:
                logger.error(f"Error loading templates from {self.template_path}: {e}")
                raise

        return self._templates

    def _is_valid_entry(self, entry: Any) -> bool:
        """Check if template entry is valid."""
        try:
            sn_info = entry.get('snInfo', None)
            return (isinstance(sn_info, np.ndarray) and
                   sn_info.shape and
                   len(sn_info.shape) == 2 and
                   sn_info.shape[0] > 0 and
                   sn_info.shape[1] == 4)
        except Exception:
            return False


class TransformerSpectrumTemplate(SpectrumTemplateInterface):
    """
    Transformer-specific template handler.
    For now, returns empty templates as Transformer doesn't use traditional templates.
    """

    def __init__(self):
        logger.info("TransformerSpectrumTemplate initialized")

    async def get_template_spectrum(self, sn_type: str, age_bin: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get template spectrum for Transformer model (not supported)."""
        raise NotImplementedError("Transformer model doesn't use traditional templates")

    async def get_all_templates(self) -> Dict[str, Any]:
        """Get all Transformer templates (empty for now)."""
        return {}

    async def validate_template(self, sn_type: str, age_bin: str) -> bool:
        """Validate Transformer template (always False)."""
        return False


def create_spectrum_template_handler(model_type: str, template_path: Optional[str] = None) -> SpectrumTemplateInterface:
    """
    Factory function to create appropriate template handler.

    Args:
        model_type: Type of model ('dash', 'transformer')
        template_path: Path to template file (required for DASH)

    Returns:
        Appropriate template handler instance
    """
    if model_type == 'dash':
        if not template_path:
            # Default template path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.join(current_dir, '..', '..', '..', '..', 'backend')
            template_path = os.path.join(backend_dir, 'astrodash_models', 'sn_and_host_templates.npz')

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        return DASHSpectrumTemplate(template_path)

    elif model_type == 'transformer':
        return TransformerSpectrumTemplate()

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
