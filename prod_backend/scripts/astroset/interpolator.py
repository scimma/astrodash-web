"""
interpolator.py
---------------
Functions for interpolating spectra to a fixed wavelength grid for the Astrodash training set pipeline.
"""

import numpy as np

def interpolate_spectrum(wavelength, flux, config):
    """
    Interpolate a spectrum to a log-wavelength grid from w0 to w1 with nw points, matching Astrodash logic.
    Args:
        wavelength (array-like): Original wavelength array.
        flux (array-like): Original flux array.
        config (dict): Configuration dictionary.
    Returns:
        np.ndarray: Interpolated flux array of length nw.
    """
    w0 = config['w0']
    w1 = config['w1']
    nw = config['nw']
    dwlog = np.log(w1 / w0) / nw
    log_wave = w0 * np.exp(np.arange(nw) * dwlog)
    interp_flux = np.interp(log_wave, wavelength, flux)
    return interp_flux
