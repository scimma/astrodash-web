# Utility functions for spectrum processing, template handling, and common operations shared across backend services.
# Place all reusable logic here to avoid duplication and improve maintainability.
import os
import numpy as np
import logging
from .astrodash_backend import get_training_parameters
import re
import math

def prepare_log_wavelength_and_templates(processed_data, template_filename='sn_and_host_templates.npz'):
    """
    Utility to prepare log-wavelength grid, interpolate input spectrum, and load templates.
    Returns: log_wave, input_flux_log, snTemplates
    """
    logger = logging.getLogger("utils")
    pars = get_training_parameters()
    w0, w1, nw = pars['w0'], pars['w1'], pars['nw']
    dwlog = np.log(w1 / w0) / nw
    log_wave = w0 * np.exp(np.arange(nw) * dwlog)
    # Find backend root and template path
    services_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.join(services_dir, '..', '..')
    template_path = os.path.join(backend_root, 'astrodash_models', template_filename)
    data = np.load(template_path, allow_pickle=True)
    snTemplates_raw = data['snTemplates'].item()
    snTemplates = {str(k): v for k, v in snTemplates_raw.items()}
    logger.debug(f"Available sn_types: {list(snTemplates.keys())}")
    for sn_type_key in snTemplates:
        logger.debug(f"  {sn_type_key}: {list(snTemplates[sn_type_key].keys())}")
    input_flux = np.array(processed_data['y'])
    input_wave = np.array(processed_data['x'])
    input_flux_log = np.interp(log_wave, input_wave, input_flux, left=0, right=0)
    return log_wave, input_flux_log, snTemplates, dwlog, nw, w0, w1

def get_templates_for_type_age(snTemplates, sn_type, age_norm, log_wave):
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

def get_nonzero_minmax(flux):
    """Return (min_index, max_index) of nonzero flux, or (0, len(flux)-1) if all zero."""
    nonzero = np.where(flux != 0)[0]
    if len(nonzero) > 0:
        return nonzero[0], nonzero[-1]
    else:
        return 0, len(flux) - 1

def normalize_age_bin(age):
    """Normalize age bin strings to 'N to M' format."""
    age = age.replace('–', '-').replace('to', '-').replace('TO', '-').replace('To', '-')
    age = age.replace(' ', '')
    match = re.match(r'(-?\d+)-(-?\d+)', age)
    if match:
        return f"{int(match.group(1))} to {int(match.group(2))}"
    return age

def get_redshift_axis(nw, dwlog):
    """Return redshift axis array for given nw and dwlog."""
    zAxisIndex = np.concatenate((np.arange(-nw // 2, 0), np.arange(0, nw // 2)))
    zAxis = np.zeros(nw)
    zAxis[:nw // 2 - 1] = -(np.exp(np.abs(zAxisIndex[:nw // 2 - 1]) * dwlog) - 1)
    zAxis[nw // 2:] = (np.exp(np.abs(zAxisIndex[nw // 2:]) * dwlog) - 1)
    zAxis = zAxis[::-1]
    return zAxis

def sanitize_for_json(obj, _path="root"):
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
        if isinstance(obj, np.bool_):
            print(f"DEBUG: Found numpy.bool_ at {_path} with value {obj}")
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v, f"{_path}.{k}") for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    else:
        return obj
