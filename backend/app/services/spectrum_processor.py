import numpy as np
from scipy import signal
from astropy.io import fits
import pandas as pd
import os
import sys
import json
from collections import OrderedDict
from urllib.request import urlopen, URLError
from .astrodash_backend import (
    get_training_parameters, AgeBinning, BestTypesListSingleRedshift, LoadInputSpectra,
    classification_split, combined_prob, RlapCalc, get_median_redshift, normalise_spectrum
)

class SpectrumProcessor:
    def __init__(self):
        self.supported_formats = ['.fits', '.dat', '.txt', '.lnw']
        self.pars = None
        try:
            self.pars = get_training_parameters()
        except Exception as e:
            print(f"Warning: Could not load training parameters: {e}")

    def read_file(self, file_or_ref):
        """Read spectrum data from various sources"""
        if isinstance(file_or_ref, str):
            if file_or_ref.startswith('osc-'):
                wave, flux, redshift = self.read_osc_input(file_or_ref)
                return {'x': wave, 'y': flux, 'redshift': redshift}
            else:
                raise ValueError("Invalid file reference")
        else:
            filename = file_or_ref.filename.lower()
            if filename.endswith('.fits'):
                return self._read_fits(file_or_ref)
            elif filename.endswith(('.dat', '.txt', '.lnw')):
                return self._read_text(file_or_ref)
            else:
                raise ValueError(f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}")

    def _read_fits(self, file):
        """Read spectrum data from FITS file"""
        try:
            with fits.open(file) as hdul:
                data = None
                for hdu in hdul:
                    if hdu.data is not None and ('wavelength' in hdu.data.names or 'WAVE' in hdu.data.names):
                        data = hdu.data
                        break
                if data is None:
                    raise ValueError("No suitable data HDU found in FITS file.")
                wavelength_col = next((col for col in ['wavelength', 'WAVE', 'LAMBDA'] if col in data.names), None)
                flux_col = next((col for col in ['flux', 'FLUX', 'SPEC'] if col in data.names), None)
                if not wavelength_col or not flux_col:
                    raise ValueError("Could not find wavelength or flux columns in FITS data.")
                wavelength = data[wavelength_col]
                flux = data[flux_col]
                return {'x': wavelength.tolist(), 'y': flux.tolist()}
        except Exception as e:
            raise ValueError(f"Error reading FITS file: {str(e)}")

    def _read_text(self, file):
        """Read spectrum data from text file (.dat, .txt, .lnw)"""
        try:
            filename = file.filename.lower() if hasattr(file, 'filename') else str(file)
            if filename.endswith('.lnw'):
                # Robust SNID .lnw parser
                import io
                import re
                # Read file content as text
                if hasattr(file, 'read'):
                    file.seek(0)
                    content = file.read().decode('utf-8') if isinstance(file.read(0), bytes) else file.read()
                    file.seek(0)
                else:
                    with open(file, 'r') as f:
                        content = f.read()
                lines = content.splitlines()
                spectrum = []
                for line in lines:
                    # Skip empty or comment lines
                    if not line.strip() or line.strip().startswith('#'):
                        continue
                    # Split by whitespace
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) < 2:
                        continue
                    try:
                        w = float(parts[0])
                        f = float(parts[1])
                        # Only accept lines where wavelength is in a reasonable range
                        if 2000 <= w <= 11000:
                            spectrum.append((w, f))
                    except Exception:
                        continue
                if not spectrum:
                    raise ValueError("No valid spectrum data found in .lnw file.")
                wavelength, flux = zip(*spectrum)
                return {'x': list(wavelength), 'y': list(flux)}
            else:
                # Always reset file pointer before reading
                if hasattr(file, 'seek'):
                    file.seek(0)
                df = pd.read_csv(file, sep=None, engine='python', header=None)
                if len(df.columns) >= 2:
                    wavelength = df.iloc[:, 0].to_numpy()
                    flux = df.iloc[:, 1].to_numpy()
                    return {'x': wavelength.tolist(), 'y': flux.tolist()}
                else:
                    raise ValueError("Text file must contain at least two columns")
        except Exception as e:
            raise ValueError(f"Error reading text file: {str(e)}")

    def read_osc_input(self, filename, template=False):
        osc_base_url = "https://api.astrocats.space/"
        obj_name = filename.split('-')[1]
        try:
            response = urlopen(f"{osc_base_url}{obj_name}/spectra/time+data")
            data = json.loads(response.read(), object_pairs_hook=OrderedDict)
            spectrum_data = data[next(iter(data))]['spectra'][0][1]
            wave, flux = np.array(spectrum_data).T.astype(float)
            return wave, flux, 0.0 # Redshift needs to be fetched separately
        except URLError as e:
            raise RuntimeError(f"Could not fetch OSC spectrum for '{filename}': {e.reason if hasattr(e, 'reason') else e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error fetching OSC spectrum for '{filename}': {e}")

    def process(self, x, y, smoothing, known_z, z_value, min_wave, max_wave, calculate_rlap):
        """Process spectrum data with given parameters using astrodash_backend logic"""
        wavelength = np.array(x)
        flux = np.array(y)

        if min_wave is not None and max_wave is not None:
            mask = (wavelength >= min_wave) & (wavelength <= max_wave)
            wavelength = wavelength[mask]
            flux = flux[mask]
            if len(wavelength) == 0:
                raise ValueError("No spectrum data within the specified wavelength range.")

        flux = self.normalize_spectrum(wavelength, flux)

        if smoothing > 0 and len(flux) > smoothing * 2:
            window_length = smoothing * 2 + 1
            if window_length > len(flux):
                window_length = len(flux) if len(flux) % 2 == 1 else len(flux) - 1
            if window_length < 3:
                window_length = 3
            polyorder = 3
            if polyorder >= window_length:
                polyorder = window_length - 1
            flux = signal.savgol_filter(flux, window_length, polyorder)

        redshift = self.calculate_redshift(wavelength, flux, known_z, z_value)

        rlap_score = None
        if calculate_rlap:
            rlap_score = self._calculate_rlap_score(flux)

        return {
            'x': wavelength.tolist(),
            'y': flux.tolist(),
            'redshift': redshift,
            'rlap_score': rlap_score,
            'calculate_rlap': calculate_rlap,
            'known_z': known_z
        }

    def normalize_spectrum(self, wavelength, flux):
        """Normalize spectrum data using astrodash's normalise_spectrum function."""
        return normalise_spectrum(flux)

    def calculate_redshift(self, wavelength, flux, known_z, z_value):
        """Calculate or set the redshift for the spectrum."""
        if known_z:
            return float(z_value)

        # For now, default to 0
        return 0.0

    def _calculate_rlap_score(self, flux):
        """A helper to calculate a simplified RLAP score."""
        # This is a placeholder for a more complex RLAP calculation.
        # It would involve fetching appropriate templates for comparison.
        return 8.0  # Placeholder value
