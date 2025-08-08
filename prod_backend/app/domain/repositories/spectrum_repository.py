from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
from app.domain.models.spectrum import Spectrum
from app.config.settings import get_settings
from app.config.logging import get_logger
from app.core.exceptions import (
    TemplateNotFoundException,
    FileNotFoundException,
    ModelConfigurationException
)
import numpy as np
import os

logger = get_logger(__name__)

class SpectrumRepository(ABC):
    """Abstract base class for spectrum repositories."""

    @abstractmethod
    async def save(self, spectrum: Spectrum) -> Spectrum:
        """Save a spectrum."""
        pass

    @abstractmethod
    async def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        """Get a spectrum by ID."""
        pass

    @abstractmethod
    async def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        """Get a spectrum by OSC reference."""
        pass

    @abstractmethod
    async def get_from_file(self, file: Any) -> Optional[Spectrum]:
        """Get a spectrum from a file."""
        pass


class SpectrumTemplateInterface(ABC):
    """Abstract interface for spectrum template handlers."""

    @abstractmethod
    async def get_template_spectrum(self, sn_type: str, age_bin: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get template spectrum for given SN type and age bin."""
        pass

    @abstractmethod
    async def get_all_templates(self) -> Dict[str, Any]:
        """Get all available templates."""
        pass

    @abstractmethod
    async def validate_template(self, sn_type: str, age_bin: str) -> bool:
        """Validate if template exists for given SN type and age bin."""
        pass


class DASHSpectrumTemplate(SpectrumTemplateInterface):
    """DASH-specific template handler."""

    def __init__(self, template_path: str):
        self.template_path = template_path
        self._templates: Optional[Dict[str, Any]] = None
        logger.info(f"DASHSpectrumTemplate initialized with path: {template_path}")

    async def get_template_spectrum(self, sn_type: str, age_bin: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get template spectrum for DASH model."""
        try:
            templates = await self._load_templates()

            if sn_type not in templates:
                raise TemplateNotFoundException(sn_type)

            if age_bin not in templates[sn_type]:
                raise TemplateNotFoundException(sn_type, age_bin)

            entry = templates[sn_type][age_bin]
            sn_info = entry.get('snInfo', None)

            if not isinstance(sn_info, np.ndarray) or sn_info.shape[0] == 0:
                raise TemplateNotFoundException(sn_type, age_bin)

            # Get the first template (placeholder for now)
            template = sn_info[0]
            wave = template[0]
            flux = template[1]

            logger.info(f"Template spectrum loaded for {sn_type} / {age_bin}")
            return wave, flux

        except TemplateNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error loading template spectrum: {e}")
            raise TemplateNotFoundException(sn_type, age_bin)

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
            raise FileNotFoundException(template_path)

        return DASHSpectrumTemplate(template_path)

    elif model_type == 'transformer':
        return TransformerSpectrumTemplate()

    else:
        raise ModelConfigurationException(f"Unsupported model type: {model_type}")
