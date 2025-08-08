from typing import Dict, List, Any, Optional
from app.domain.repositories.spectrum_repository import SpectrumTemplateInterface
from app.shared.utils.validators import ValidationError
from app.config.logging import get_logger
import numpy as np

logger = get_logger(__name__)

class TemplateAnalysisService:
    """
    Domain service for analyzing and validating template data.
    Handles template validation, SN type extraction, and age bin analysis.
    """

    def __init__(self, template_handler: SpectrumTemplateInterface):
        self.template_handler = template_handler
        logger.info("TemplateAnalysisService initialized")

    async def get_analysis_options(self) -> Dict[str, Any]:
        """
        Get valid SN types and age bins for analysis.

        Returns:
            Dictionary with 'sn_types' and 'age_bins_by_type'
        """
        try:
            logger.info("Getting analysis options from templates")

            # Get all available templates
            templates = await self.template_handler.get_all_templates()

            # Validate and extract valid SN types and age bins
            valid_options = self._validate_and_extract_options(templates)

            # Extract SN types list
            sn_types = list(valid_options.keys())

            logger.info(f"Found {len(sn_types)} valid SN types: {sn_types}")

            return {
                'sn_types': sn_types,
                'age_bins_by_type': valid_options
            }

        except Exception as e:
            logger.error(f"Error getting analysis options: {e}")
            raise ValidationError(f"Failed to get analysis options: {str(e)}")

    def _validate_and_extract_options(self, templates: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate templates and extract valid SN types and age bins.

        Args:
            templates: Raw template data from repository

        Returns:
            Dictionary mapping SN types to valid age bins
        """
        valid_options = {}

        for sn_type, age_bins in templates.items():
            valid_bins = []

            for age_bin, entry in age_bins.items():
                # Validate template entry
                if self._is_valid_template_entry(entry):
                    valid_bins.append(age_bin)

            # Only include SN types that have valid age bins
            if valid_bins:
                valid_options[sn_type] = valid_bins
                logger.debug(f"Valid SN type '{sn_type}' with {len(valid_bins)} age bins")

        return valid_options

    def _is_valid_template_entry(self, entry: Any) -> bool:
        """
        Validate if a template entry has valid snInfo data.

        Args:
            entry: Template entry to validate

        Returns:
            True if entry is valid, False otherwise
        """
        try:
            # Check if entry has snInfo
            sn_info = entry.get('snInfo', None)

            # Validate snInfo structure
            if not isinstance(sn_info, np.ndarray):
                return False

            # Check shape requirements
            if not sn_info.shape or len(sn_info.shape) != 2:
                return False

            # Check minimum size requirements
            if sn_info.shape[0] == 0:
                return False

            # Check expected column structure (wavelength, flux, etc.)
            if sn_info.shape[1] != 4:  # Expected: [wavelength, flux, age, type]
                return False

            return True

        except Exception as e:
            logger.debug(f"Template entry validation failed: {e}")
            return False

    async def get_template_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available templates.

        Returns:
            Dictionary with template statistics
        """
        try:
            templates = await self.template_handler.get_all_templates()

            total_sn_types = len(templates)
            total_age_bins = sum(len(age_bins) for age_bins in templates.values())
            valid_sn_types = 0
            valid_age_bins = 0

            for sn_type, age_bins in templates.items():
                type_valid_bins = 0
                for age_bin, entry in age_bins.items():
                    if self._is_valid_template_entry(entry):
                        type_valid_bins += 1

                if type_valid_bins > 0:
                    valid_sn_types += 1
                    valid_age_bins += type_valid_bins

            return {
                'total_sn_types': total_sn_types,
                'total_age_bins': total_age_bins,
                'valid_sn_types': valid_sn_types,
                'valid_age_bins': valid_age_bins,
                'validation_rate': {
                    'sn_types': valid_sn_types / total_sn_types if total_sn_types > 0 else 0,
                    'age_bins': valid_age_bins / total_age_bins if total_age_bins > 0 else 0
                }
            }

        except Exception as e:
            logger.error(f"Error getting template statistics: {e}")
            raise ValidationError(f"Failed to get template statistics: {str(e)}")

    async def validate_template_request(self, sn_type: str, age_bin: str) -> bool:
        """
        Validate if a specific SN type and age bin combination is available.

        Args:
            sn_type: Supernova type to validate
            age_bin: Age bin to validate

        Returns:
            True if combination is valid, False otherwise
        """
        try:
            # Get analysis options
            options = await self.get_analysis_options()

            # Check if SN type exists
            if sn_type not in options['age_bins_by_type']:
                logger.warning(f"SN type '{sn_type}' not found in valid options")
                return False

            # Check if age bin exists for this SN type
            valid_age_bins = options['age_bins_by_type'][sn_type]
            if age_bin not in valid_age_bins:
                logger.warning(f"Age bin '{age_bin}' not found for SN type '{sn_type}'")
                return False

            logger.debug(f"Template request validated: {sn_type} / {age_bin}")
            return True

        except Exception as e:
            logger.error(f"Error validating template request: {e}")
            return False
