import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LineListService:
    """Service for handling element/ion line list operations."""

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the LineListService.

        Args:
            models_dir: Directory containing the line list file. If None, uses default location.
        """
        if models_dir is None:
            # Default to astrodash_models directory relative to this service
            services_dir = os.path.dirname(os.path.abspath(__file__))
            backend_root = os.path.join(services_dir, '..', '..')
            models_dir = os.path.join(backend_root, 'astrodash_models')

        self.models_dir = models_dir
        self.line_list_path = os.path.join(models_dir, 'sneLineList.txt')
        self._line_list_cache: Optional[Dict[str, List[float]]] = None

        logger.info(f"LineListService initialized with models directory: {models_dir}")
        logger.info(f"Line list file path: {self.line_list_path}")

    def load_line_list(self) -> Dict[str, List[float]]:
        """
        Load and parse the element/ion line list from the sneLineList.txt file.

        Returns:
            Dictionary mapping element/ion names to lists of wavelengths.

        Raises:
            FileNotFoundError: If the line list file doesn't exist.
            ValueError: If the file format is invalid.
        """
        if self._line_list_cache is not None:
            logger.debug("Returning cached line list")
            return self._line_list_cache

        if not os.path.exists(self.line_list_path):
            logger.error(f"Line list file not found: {self.line_list_path}")
            raise FileNotFoundError(f"Line list file not found: {self.line_list_path}")

        try:
            line_dict = {}
            raw_lines = []

            with open(self.line_list_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    raw_lines.append(line.rstrip())
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Parse line in format "Element: wavelength1, wavelength2, ..."
                    if ':' not in line:
                        logger.warning(f"Skipping invalid line {line_num}: {line}")
                        continue

                    try:
                        key, values = line.split(':', 1)
                        key = key.strip()

                        # Parse wavelengths, handling various separators
                        wavelength_str = values.replace(',', ' ').replace(';', ' ')
                        wavelengths = []

                        for w_str in wavelength_str.split():
                            w_str = w_str.strip()
                            if w_str:  # Skip empty strings
                                try:
                                    wavelength = float(w_str)
                                    wavelengths.append(wavelength)
                                except ValueError:
                                    logger.warning(f"Invalid wavelength '{w_str}' in line {line_num}")

                        if wavelengths:
                            line_dict[key] = wavelengths
                        else:
                            logger.warning(f"No valid wavelengths found for {key} in line {line_num}")

                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {line}. Error: {e}")
                        continue

            logger.info(f"Successfully loaded line list with {len(line_dict)} elements/ions")
            logger.debug(f"Raw sneLineList.txt lines: {raw_lines}")
            logger.debug(f"Parsed line_dict: {line_dict}")

            # Cache the result
            self._line_list_cache = line_dict
            return line_dict

        except Exception as e:
            logger.error(f"Error reading line list file: {e}", exc_info=True)
            raise ValueError(f"Error reading line list file: {e}")

    def get_line_list(self) -> Dict[str, List[float]]:
        """
        Get the line list, loading it if necessary.

        Returns:
            Dictionary mapping element/ion names to lists of wavelengths.
        """
        return self.load_line_list()

    def get_element_wavelengths(self, element: str) -> List[float]:
        """
        Get wavelengths for a specific element/ion.

        Args:
            element: Name of the element/ion (e.g., 'H_Balmer', 'HeI')

        Returns:
            List of wavelengths for the specified element/ion.

        Raises:
            KeyError: If the element is not found in the line list.
        """
        line_list = self.get_line_list()
        if element not in line_list:
            raise KeyError(f"Element '{element}' not found in line list")
        return line_list[element]

    def get_available_elements(self) -> List[str]:
        """
        Get list of all available elements/ions in the line list.

        Returns:
            List of element/ion names.
        """
        line_list = self.get_line_list()
        return list(line_list.keys())

    def filter_wavelengths_by_range(self, min_wavelength: float, max_wavelength: float) -> Dict[str, List[float]]:
        """
        Filter the line list to only include wavelengths within a specified range.

        Args:
            min_wavelength: Minimum wavelength (inclusive)
            max_wavelength: Maximum wavelength (inclusive)

        Returns:
            Filtered dictionary with only wavelengths in the specified range.
        """
        line_list = self.get_line_list()
        filtered_dict = {}

        for element, wavelengths in line_list.items():
            filtered_wavelengths = [
                w for w in wavelengths
                if min_wavelength <= w <= max_wavelength
            ]
            if filtered_wavelengths:
                filtered_dict[element] = filtered_wavelengths

        logger.info(f"Filtered line list to {len(filtered_dict)} elements with wavelengths in range [{min_wavelength}, {max_wavelength}]")
        return filtered_dict

    def clear_cache(self):
        """Clear the internal cache, forcing reload on next access."""
        self._line_list_cache = None
        logger.debug("Line list cache cleared")


# Global instance for easy access
line_list_service = LineListService()
