from abc import ABC, abstractmethod
from typing import Optional, Any, List, Tuple, Dict
import numpy as np
import logging
from domain.models.spectrum import Spectrum

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
    Abstract interface for spectrum template handling.
    Different implementations can handle different template formats and sources.
    """

    @abstractmethod
    def load_template_spectrum(self, sn_type: str, age_bin: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single template spectrum for given SN type and age bin.

        Args:
            sn_type: Supernova type (e.g., 'Ia', 'Ib', 'II')
            age_bin: Age bin string (e.g., '0 to 5')
            **kwargs: Additional parameters specific to implementation

        Returns:
            Tuple of (wavelength_array, flux_array)

        Raises:
            ValueError: If template not found or invalid
        """
        pass

    @abstractmethod
    def get_templates_for_type_age(
        self,
        sn_type: str,
        age_bin: str,
        wavelength_grid: np.ndarray,
        **kwargs
    ) -> Tuple[List[np.ndarray], List[str], List[Tuple[int, int]]]:
        """
        Get multiple template spectra for given SN type and age bin, interpolated to wavelength grid.

        Args:
            sn_type: Supernova type
            age_bin: Age bin string
            wavelength_grid: Target wavelength grid for interpolation
            **kwargs: Additional parameters specific to implementation

        Returns:
            Tuple of (template_fluxes, template_names, template_minmax_indexes)
        """
        pass

    @abstractmethod
    def get_valid_sn_types_and_age_bins(self, **kwargs) -> Dict[str, List[str]]:
        """
        Get all valid SN types and their corresponding age bins.

        Args:
            **kwargs: Additional parameters specific to implementation

        Returns:
            Dictionary mapping SN types to lists of valid age bins
        """
        pass


class DASHSpectrumTemplate(SpectrumTemplateInterface):
    """
    DASH-specific implementation of spectrum template handling.
    Works with DASH's nested template structure in .npz files.
    """

    def __init__(self, npz_path: str):
        """
        Initialize with path to DASH template file.

        Args:
            npz_path: Path to .npz file containing DASH templates
        """
        self.npz_path = npz_path
        self._templates_cache = None

    def _load_templates(self) -> Dict[str, Any]:
        """Load and cache templates from npz file."""
        if self._templates_cache is None:
            data = np.load(self.npz_path, allow_pickle=True)
            snTemplates_raw = data['snTemplates'].item() if 'snTemplates' in data else data['arr_0'].item()
            self._templates_cache = {str(k): v for k, v in snTemplates_raw.items()}
        return self._templates_cache

    def load_template_spectrum(self, sn_type: str, age_bin: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single template spectrum for given SN type and age bin.

        Args:
            sn_type: Supernova type (e.g., 'Ia', 'Ib', 'II')
            age_bin: Age bin string (e.g., '0 to 5')
            **kwargs: Additional parameters (not used in DASH implementation)

        Returns:
            Tuple of (wavelength_array, flux_array)

        Raises:
            ValueError: If template not found or invalid
        """
        logger.info(f"Loading template spectrum for SN type {sn_type}, age bin {age_bin}")

        snTemplates = self._load_templates()

        if not isinstance(snTemplates[sn_type], dict):
            snTemplates[sn_type] = dict(snTemplates[sn_type])

        if age_bin not in snTemplates[sn_type].keys():
            logger.error(f"Age bin '{age_bin}' not found for SN type '{sn_type}'.")
            raise ValueError(f"Age bin '{age_bin}' not found for SN type '{sn_type}'.")

        snInfo = snTemplates[sn_type][age_bin].get('snInfo', None)
        if not isinstance(snInfo, np.ndarray) or snInfo.shape[0] == 0:
            logger.error(f"No template spectrum available for SN type '{sn_type}' and age bin '{age_bin}'.")
            raise ValueError(f"No template spectrum available for SN type '{sn_type}' and age bin '{age_bin}'.")

        template = snInfo[0]  # placeholder for now
        wave = template[0]
        flux = template[1]

        logger.info(f"Template spectrum loaded for {sn_type} / {age_bin}")
        return wave, flux

    def get_templates_for_type_age(
        self,
        sn_type: str,
        age_bin: str,
        wavelength_grid: np.ndarray,
        **kwargs
    ) -> Tuple[List[np.ndarray], List[str], List[Tuple[int, int]]]:
        """
        Get multiple template spectra for given SN type and age bin.

        This implementation handles DASH's nested template structure where multiple
        templates can exist for a single type/age combination.

        Args:
            sn_type: Supernova type
            age_bin: Age bin string
            wavelength_grid: Target wavelength grid for interpolation
            **kwargs: Additional parameters (not used in DASH implementation)

        Returns:
            Tuple of (template_fluxes, template_names, template_minmax_indexes)
        """
        snTemplates = self._load_templates()
        template_fluxes = []
        template_names = []
        template_minmax_indexes = []

        if sn_type in snTemplates:
            age_bin_keys = [str(k).strip() for k in snTemplates[sn_type].keys()]
            if age_bin.strip() in age_bin_keys:
                real_key = [k for k in snTemplates[sn_type].keys() if str(k).strip() == age_bin.strip()][0]
                snInfo = snTemplates[sn_type][real_key].get('snInfo', None)
                if isinstance(snInfo, np.ndarray) and snInfo.shape[0] > 0:
                    for i in range(snInfo.shape[0]):
                        template_wave = snInfo[i][0]
                        template_flux = snInfo[i][1]
                        interp_flux = np.interp(wavelength_grid, template_wave, template_flux, left=0, right=0)
                        nonzero = np.where(interp_flux != 0)[0]
                        if len(nonzero) > 0:
                            tmin, tmax = nonzero[0], nonzero[-1]
                        else:
                            tmin, tmax = 0, len(interp_flux) - 1
                        template_fluxes.append(interp_flux)
                        template_names.append(f"{sn_type}:{age_bin}")
                        template_minmax_indexes.append((tmin, tmax))

        return template_fluxes, template_names, template_minmax_indexes

    def get_valid_sn_types_and_age_bins(self, **kwargs) -> Dict[str, List[str]]:
        """
        Get all valid SN types and their corresponding age bins.

        Args:
            **kwargs: Additional parameters (not used in DASH implementation)

        Returns:
            Dictionary mapping SN types to lists of valid age bins
        """
        snTemplates = self._load_templates()
        valid = {}

        for sn_type, age_bins in snTemplates.items():
            valid_bins = []
            for age_bin, entry in age_bins.items():
                snInfo = entry.get('snInfo', None)
                if (
                    isinstance(snInfo, np.ndarray) and
                    snInfo.shape and
                    len(snInfo.shape) == 2 and
                    snInfo.shape[0] > 0 and
                    snInfo.shape[1] == 4
                ):
                    valid_bins.append(age_bin)
            if valid_bins:
                valid[sn_type] = valid_bins

        logger.info(f"Valid SN types and age bins loaded: {list(valid.keys())}")
        return valid





class TransformerSpectrumTemplate(SpectrumTemplateInterface):
    """
    Placeholder implementation for transformer-based spectrum templates.
    To be implemented when transformer templates are available.
    """

    def __init__(self, model_path: str):
        """
        Initialize with path to transformer model.

        Args:
            model_path: Path to transformer model
        """
        self.model_path = model_path
        logger.warning("TransformerSpectrumTemplate is a placeholder implementation")

    def load_template_spectrum(self, sn_type: str, age_bin: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Placeholder - to be implemented when transformer templates are available."""
        raise NotImplementedError("Transformer spectrum templates not yet implemented")

    def get_templates_for_type_age(
        self,
        sn_type: str,
        age_bin: str,
        wavelength_grid: np.ndarray,
        **kwargs
    ) -> Tuple[List[np.ndarray], List[str], List[Tuple[int, int]]]:
        """Placeholder - to be implemented when transformer templates are available."""
        raise NotImplementedError("Transformer spectrum templates not yet implemented")

    def get_valid_sn_types_and_age_bins(self, **kwargs) -> Dict[str, List[str]]:
        """Placeholder - to be implemented when transformer templates are available."""
        raise NotImplementedError("Transformer spectrum templates not yet implemented")


def create_spectrum_template_handler(template_type: str, **kwargs) -> SpectrumTemplateInterface:
    """
    Factory function to create appropriate spectrum template handler.

    Args:
        template_type: Type of template handler ('dash', 'transformer', etc.)
        **kwargs: Parameters specific to the template type

    Returns:
        Appropriate SpectrumTemplateInterface implementation

    Raises:
        ValueError: If template_type is not supported or required parameters missing
    """
    template_type = template_type.lower()

    if template_type == 'dash':
        if 'npz_path' not in kwargs:
            raise ValueError("DASH template handler requires 'npz_path' parameter")
        return DASHSpectrumTemplate(kwargs['npz_path'])

    elif template_type == 'transformer':
        if 'model_path' not in kwargs:
            raise ValueError("Transformer template handler requires 'model_path' parameter")
        return TransformerSpectrumTemplate(kwargs['model_path'])

    else:
        raise ValueError(f"Unsupported template type: {template_type}")
