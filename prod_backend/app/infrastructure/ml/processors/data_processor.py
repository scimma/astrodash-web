import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import splrep, splev
from typing import Tuple
import logging

logger = logging.getLogger("dash_processor")

class DashSpectrumProcessor:
    """
    Handles all preprocessing for the Dash (CNN) classifier.
    Includes normalization, wavelength binning, continuum removal, mean zeroing, and apodization.
    """
    def __init__(self, w0: float, w1: float, nw: int, num_spline_points: int = 13):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.num_spline_points = num_spline_points

    def process(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        z: float,
        smooth: int = 0,
        min_wave: float = None,
        max_wave: float = None
    ) -> Tuple[np.ndarray, int, int, float]:
        """
        Full preprocessing pipeline for Dash classifier.
        Returns: (processed_flux, min_idx, max_idx, z)
        """
        flux = self.normalise_spectrum(flux)
        flux = self.limit_wavelength_range(wave, flux, min_wave, max_wave)
        if smooth > 0:
            wavelength_density = (np.max(wave) - np.min(wave)) / len(wave)
            w_density = (self.w1 - self.w0) / self.nw
            filter_size = int(w_density / wavelength_density * smooth / 2) * 2 + 1
            if filter_size >= 3:
                flux = medfilt(flux, kernel_size=filter_size)
        wave_deredshifted = wave / (1 + z)
        if len(wave_deredshifted) < 2:
            logger.error("Spectrum is out of classification range after deredshifting.")
            raise ValueError("Spectrum is out of classification range.")
        binned_wave, binned_flux, min_idx, max_idx = self.log_wavelength_binning(wave_deredshifted, flux)
        new_flux, _ = self.continuum_removal(binned_wave, binned_flux, min_idx, max_idx)
        mean_zero_flux = self.mean_zero(new_flux, min_idx, max_idx)
        apodized_flux = self.apodize(mean_zero_flux, min_idx, max_idx)
        flux_norm = self.normalise_spectrum(apodized_flux)
        flux_norm = self.zero_non_overlap_part(flux_norm, min_idx, max_idx, outer_val=0.5)
        return flux_norm, min_idx, max_idx, z

    @staticmethod
    def normalise_spectrum(flux: np.ndarray) -> np.ndarray:
        if len(flux) == 0 or np.min(flux) == np.max(flux):
            logger.warning("Normalising spectrum: zero or constant flux array.")
            return np.zeros(len(flux))
        return (flux - np.min(flux)) / (np.max(flux) - np.min(flux))

    @staticmethod
    def limit_wavelength_range(wave: np.ndarray, flux: np.ndarray, min_wave: float, max_wave: float) -> np.ndarray:
        if min_wave is not None:
            min_idx = (np.abs(wave - min_wave)).argmin()
            flux[:min_idx] = 0
        if max_wave is not None:
            max_idx = (np.abs(wave - max_wave)).argmin()
            flux[max_idx:] = 0
        return flux

    def log_wavelength_binning(self, wave: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        dwlog = np.log(self.w1 / self.w0) / self.nw
        wlog = self.w0 * np.exp(np.arange(0, self.nw) * dwlog)
        binned_flux = np.interp(wlog, wave, flux, left=0, right=0)
        non_zero_indices = np.where(binned_flux != 0)[0]
        min_index = non_zero_indices[0] if len(non_zero_indices) > 0 else 0
        max_index = non_zero_indices[-1] if len(non_zero_indices) > 0 else self.nw - 1
        return wlog, binned_flux, min_index, max_index

    def continuum_removal(self, wave: np.ndarray, flux: np.ndarray, min_idx: int, max_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        wave_region, flux_region = wave[min_idx:max_idx+1], flux[min_idx:max_idx+1]
        if len(wave_region) > self.num_spline_points:
            indices = np.linspace(0, len(wave_region)-1, self.num_spline_points, dtype=int)
            spline = splrep(wave_region[indices], flux_region[indices], k=3)
            continuum = splev(wave_region, spline)
        else:
            continuum = np.mean(flux_region)
        full_continuum = np.zeros_like(flux)
        full_continuum[min_idx:max_idx+1] = continuum
        return flux - full_continuum, full_continuum

    @staticmethod
    def mean_zero(flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        flux_out = np.copy(flux)
        flux_out[0:min_idx] = flux_out[min_idx]
        flux_out[max_idx:] = flux_out[min_idx]
        return flux_out - flux_out[min_idx]

    @staticmethod
    def apodize(flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        apodized = np.copy(flux)
        edge_width = min(50, (max_idx - min_idx) // 4)
        if edge_width > 0:
            for i in range(edge_width):
                factor = 0.5 * (1 + np.cos(np.pi * i / edge_width))
                if min_idx + i < len(apodized): apodized[min_idx + i] *= factor
                if max_idx - i >= 0: apodized[max_idx - i] *= factor
        return apodized

    @staticmethod
    def zero_non_overlap_part(array: np.ndarray, min_index: int, max_index: int, outer_val: float = 0.0) -> np.ndarray:
        sliced_array = np.copy(array)
        sliced_array[0:min_index] = outer_val
        sliced_array[max_index:] = outer_val
        return sliced_array

class TransformerSpectrumProcessor:
    """
    Handles preprocessing for the Transformer classifier.
    Includes interpolation to 1024 points and normalization.
    """
    def __init__(self, target_length: int = 1024):
        self.target_length = target_length

    def process(self, x, y, redshift: float = 0.0):
        """
        Interpolates x and y to target_length, normalizes y, and returns processed arrays.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_interp = self._interpolate_to_length(x, self.target_length)
        y_interp = self._interpolate_to_length(y, self.target_length)
        y_norm = self._normalize(y_interp)
        return x_interp, y_norm, redshift

    def _interpolate_to_length(self, arr, length):
        arr = np.asarray(arr)
        if len(arr) == length:
            return arr
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, length)
        return np.interp(x_new, x_old, arr)

    def _normalize(self, arr):
        arr = np.asarray(arr)
        if len(arr) == 0 or np.min(arr) == np.max(arr):
            logger.warning("Normalising transformer input: zero or constant array.")
            return np.zeros(len(arr))
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
