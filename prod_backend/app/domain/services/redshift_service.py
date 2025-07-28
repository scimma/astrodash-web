from typing import Any, Dict, List, Optional, Tuple
from shared.utils.redshift import get_median_redshift
import numpy as np

class RedshiftService:
    """
    Service for estimating redshift from a spectrum and templates.
    """
    def __init__(self, settings=None):
        self.settings = settings

    async def estimate_redshift(
        self,
        input_flux: np.ndarray,
        temp_fluxes: List[np.ndarray],
        nw: int,
        dwlog: float,
        input_minmax_index: Any,
        temp_minmax_indexes: List[Any],
        temp_names: List[str],
        outer_val: float = 0.5
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]], Optional[str], Optional[float]]:
        """
        Estimate the median redshift for the input spectrum using provided templates.
        Returns (median_redshift, crossCorrs, medianName, stdRedshift)
        """
        return get_median_redshift(
            input_flux, temp_fluxes, nw, dwlog, input_minmax_index, temp_minmax_indexes, temp_names, outer_val
        )
