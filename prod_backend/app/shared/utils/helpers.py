import os
import numpy as np
import re
import math
from typing import Any, Dict, List, Tuple, Optional

def prepare_log_wavelength_and_templates(processed_data: Dict[str, Any], template_filename: str = 'sn_and_host_templates.npz') -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float, int, float, float]:
    """
    Prepare log-wavelength grid, interpolate input spectrum, and load templates.
    Returns: log_wave, input_flux_log, snTemplates, dwlog, nw, w0, w1
    """
    from config.settings import get_settings
    settings = get_settings()
    pars = settings  # Or load from a config or parameter file as needed
    w0, w1, nw = 3500.0, 10000.0, 1024  # Defaults; override as needed
    dwlog = np.log(w1 / w0) / nw
    log_wave = w0 * np.exp(np.arange(nw) * dwlog)
    # Find backend root and template path
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    template_path = os.path.join(backend_root, 'astrodash_models', template_filename)
    data = np.load(template_path, allow_pickle=True)
    snTemplates_raw = data['snTemplates'].item()
    snTemplates = {str(k): v for k, v in snTemplates_raw.items()}
    input_flux = np.array(processed_data['y'])
    input_wave = np.array(processed_data['x'])
    input_flux_log = np.interp(log_wave, input_wave, input_flux, left=0, right=0)
    return log_wave, input_flux_log, snTemplates, dwlog, nw, w0, w1

def get_templates_for_type_age(snTemplates: Dict[str, Any], sn_type: str, age_norm: str, log_wave: np.ndarray) -> Tuple[List[np.ndarray], List[str], List[Tuple[int, int]]]:
    """
    Given snTemplates dict, SN type, normalized age bin, and log-wavelength grid,
    return template_fluxes, template_names, template_minmax_indexes for that type/age.
    """
    template_fluxes = []
    template_names = []
    template_minmax_indexes = []
    if sn_type in snTemplates:
        age_bin_keys = [str(k).strip() for k in snTemplates[sn_type].keys()]
        if age_norm.strip() in age_bin_keys:
            real_key = [k for k in snTemplates[sn_type].keys() if str(k).strip() == age_norm.strip()][0]
            snInfo = snTemplates[sn_type][real_key].get('snInfo', None)
            if isinstance(snInfo, np.ndarray) and snInfo.shape[0] > 0:
                for i in range(snInfo.shape[0]):
                    template_wave = snInfo[i][0]
                    template_flux = snInfo[i][1]
                    interp_flux = np.interp(log_wave, template_wave, template_flux, left=0, right=0)
                    nonzero = np.where(interp_flux != 0)[0]
                    if len(nonzero) > 0:
                        tmin, tmax = nonzero[0], nonzero[-1]
                    else:
                        tmin, tmax = 0, len(interp_flux) - 1
                    template_fluxes.append(interp_flux)
                    template_names.append(f"{sn_type}:{age_norm}")
                    template_minmax_indexes.append((tmin, tmax))
    return template_fluxes, template_names, template_minmax_indexes

def get_nonzero_minmax(flux: np.ndarray) -> Tuple[int, int]:
    """Return (min_index, max_index) of nonzero flux, or (0, len(flux)-1) if all zero."""
    nonzero = np.where(flux != 0)[0]
    if len(nonzero) > 0:
        return nonzero[0], nonzero[-1]
    else:
        return 0, len(flux) - 1

def normalize_age_bin(age: str) -> str:
    """Normalize age bin strings to 'N to M' format."""
    age = age.replace('–', '-').replace('to', '-').replace('TO', '-').replace('To', '-')
    age = age.replace(' ', '')
    match = re.match(r'(-?\d+)-(-?\d+)', age)
    if match:
        return f"{int(match.group(1))} to {int(match.group(2))}"
    return age

def sanitize_for_json(obj: Any, _path: str = "root") -> Any:
    """Sanitizes for JSON format (deals with inf values, numpy types, etc.)"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v, f"{_path}.{k}") for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    else:
        return obj

def get_redshift_axis(nw: int, dwlog: float) -> np.ndarray:
    """Return redshift axis array for given nw and dwlog."""
    zAxisIndex = np.concatenate((np.arange(-nw // 2, 0), np.arange(0, nw // 2)))
    zAxis = np.zeros(nw)
    zAxis[:nw // 2 - 1] = -(np.exp(np.abs(zAxisIndex[:nw // 2 - 1]) * dwlog) - 1)
    zAxis[nw // 2:] = (np.exp(np.abs(zAxisIndex[nw // 2:]) * dwlog) - 1)
    zAxis = zAxis[::-1]
    return zAxis

def mean_zero_spectra(flux: np.ndarray, min_idx: int, max_idx: int, nw: int) -> np.ndarray:
    """Mean-zero a region of a spectrum."""
    out = np.zeros(nw)
    region = flux[min_idx:max_idx+1]
    mean = np.mean(region) if len(region) > 0 else 0
    out[min_idx:max_idx+1] = region - mean
    return out

def normalise_spectrum(flux: np.ndarray) -> np.ndarray:
    """Normalize a spectrum to [0, 1] range."""
    if len(flux) == 0 or np.min(flux) == np.max(flux):
        return np.zeros(len(flux))
    return (flux - np.min(flux)) / (np.max(flux) - np.min(flux))

def zero_non_overlap_part(array: np.ndarray, min_index: int, max_index: int, outer_val: float = 0.0) -> np.ndarray:
    """Zero out (or set to outer_val) the non-overlap part of an array."""
    sliced_array = np.copy(array)
    sliced_array[0:min_index] = outer_val
    sliced_array[max_index:] = outer_val
    return sliced_array

def limit_wavelength_range(wave: np.ndarray, flux: np.ndarray, min_wave: Optional[float], max_wave: Optional[float]) -> np.ndarray:
    """Zero out flux outside the [min_wave, max_wave] range."""
    if min_wave is not None:
        min_idx = (np.abs(wave - min_wave)).argmin()
        flux[:min_idx] = 0
    if max_wave is not None:
        max_idx = (np.abs(wave - max_wave)).argmin()
        flux[max_idx:] = 0
    return flux

def shift_to_rest_frame(wave: np.ndarray, flux: np.ndarray, redshift: float) -> Tuple[np.ndarray, np.ndarray]:
    """Shift observed spectrum to rest-frame using the given redshift."""
    rest_wave = wave / (1 + redshift)
    return rest_wave, flux
