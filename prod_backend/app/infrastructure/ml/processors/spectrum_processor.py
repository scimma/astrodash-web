import numpy as np
from scipy import signal
from astropy.io import fits
import pandas as pd
import os
import sys
import json
from collections import OrderedDict
from urllib.request import urlopen
from urllib.error import URLError
# NOTE: You will need to adapt the following import to your new architecture
# from .astrodash_backend import (
#     get_training_parameters, AgeBinning, BestTypesListSingleRedshift, LoadInputSpectra,
#     classification_split, combined_prob, normalise_spectrum
# )
import logging
logger = logging.getLogger("spectrum_processor")

class SpectrumProcessor:
    def __init__(self):
        self.supported_formats = ['.fits', '.dat', '.txt', '.lnw']
        self.pars = None
        # TODO: Adapt or implement get_training_parameters and normalise_spectrum in new codebase
        self.pars = None

    def read_file(self, file_or_ref):
        """Read spectrum data from various sources"""
        if isinstance(file_or_ref, str):
            if file_or_ref.startswith('osc-'):
                logger.info(f"Reading OSC input: {file_or_ref}")
                wave, flux, redshift = self.read_osc_input(file_or_ref)
                return {'x': wave, 'y': flux, 'redshift': redshift}
            else:
                logger.error("Invalid file reference string.")
                raise ValueError("Invalid file reference")
        else:
            filename = None
            file_obj = file_or_ref
            if hasattr(file_or_ref, 'filename') and hasattr(file_or_ref, 'file'):
                filename = file_or_ref.filename.lower()
                file_obj = file_or_ref.file
            elif hasattr(file_or_ref, 'name'):
                filename = os.path.basename(file_or_ref.name).lower()
            if not filename:
                logger.error("Could not determine filename or file type for uploaded file.")
                raise ValueError("Could not determine filename or file type for uploaded file.")
            logger.info(f"Reading file: {filename}")
            if filename.endswith('.fits'):
                return self._read_fits(file_obj)
            elif filename.endswith(('.dat', '.txt', '.lnw')):
                return self._read_text(file_obj, filename)
            else:
                logger.error(f"Unsupported file format: {filename}")
                raise ValueError(f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}")

    def _read_fits(self, file):
        try:
            with fits.open(file) as hdul:
                data = None
                for hdu in hdul:  # type: ignore
                    try:
                        if hdu.data is not None and hasattr(hdu.data, 'names'):  # type: ignore
                            if 'wavelength' in hdu.data.names or 'WAVE' in hdu.data.names:  # type: ignore
                                data = hdu.data  # type: ignore
                                break
                    except AttributeError:
                        continue
                if data is None:
                    logger.error("No suitable data HDU found in FITS file.")
                    raise ValueError("No suitable data HDU found in FITS file.")
                wavelength_col = next((col for col in ['wavelength', 'WAVE', 'LAMBDA'] if col in data.names), None)
                flux_col = next((col for col in ['flux', 'FLUX', 'SPEC'] if col in data.names), None)
                if not wavelength_col or not flux_col:
                    logger.error("Could not find wavelength or flux columns in FITS data.")
                    raise ValueError("Could not find wavelength or flux columns in FITS data.")
                wavelength = data[wavelength_col]
                flux = data[flux_col]
                logger.info("Successfully read FITS file.")
                return {'x': wavelength.tolist(), 'y': flux.tolist()}
        except Exception as e:
            logger.error(f"Error reading FITS file: {str(e)}")
            raise ValueError(f"Error reading FITS file: {str(e)}")

    def _read_text(self, file, filename):
        try:
            if filename.endswith('.lnw'):
                import re
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
                    if not line.strip() or line.strip().startswith('#'):
                        continue
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) < 2:
                        continue
                    try:
                        w = float(parts[0])
                        f = float(parts[1])
                        if 2000 <= w <= 11000:
                            spectrum.append((w, f))
                    except Exception:
                        continue
                if not spectrum:
                    logger.error(f"No valid spectrum data found in .lnw file: {filename}")
                    raise ValueError("No valid spectrum data found in .lnw file.")
                wavelength, flux = zip(*spectrum)
                logger.info(f"Successfully read .lnw file: {filename}")
                return {'x': list(wavelength), 'y': list(flux)}
            else:
                if hasattr(file, 'seek'):
                    file.seek(0)
                if hasattr(file, 'read'):
                    content = file.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                    from io import StringIO
                    file_obj = StringIO(content)
                else:
                    file_obj = file
                file_obj.seek(0)
                preview = ''.join([file_obj.readline() for _ in range(2)])
                logger.debug(f"First 2 lines of text file {filename}:\n{preview}")
                file_obj.seek(0)
                df = pd.read_csv(file_obj, sep='\s+', header=None, comment='#')
                logger.debug(f"DataFrame head for {filename}:\n{df.head()}")
                if len(df.columns) >= 2:
                    wavelength = df.iloc[:, 0].to_numpy()
                    flux = df.iloc[:, 1].to_numpy()
                    logger.info(f"Successfully read text file: {filename}")
                    return {'x': wavelength.tolist(), 'y': flux.tolist()}
                else:
                    logger.error(f"Text file must contain at least two columns: {filename}")
                    raise ValueError("Text file must contain at least two columns")
        except Exception as e:
            logger.error(f"Error reading text file {filename}: {str(e)}")
            raise ValueError(f"Error reading text file: {str(e)}")

    def read_osc_input(self, filename):
        osc_base_url = "https://api.astrocats.space/"
        obj_name = filename.split('-')[1]
        try:
            import ssl
            context = ssl._create_unverified_context()
            from urllib.request import urlopen
            response = urlopen(f"{osc_base_url}{obj_name}/spectra/time+data", context=context)
            data = json.loads(response.read(), object_pairs_hook=OrderedDict)
            spectrum_data = data[next(iter(data))]['spectra'][0][1]
            wave, flux = np.array(spectrum_data).T.astype(float)
            logger.info(f"Successfully fetched OSC spectrum for {filename}")
            return wave, flux, 0.0 # Redshift needs to be fetched separately
        except URLError as e:
            logger.error(f"Could not fetch OSC spectrum for '{filename}': {e.reason if hasattr(e, 'reason') else e}")
            raise RuntimeError(f"Could not fetch OSC spectrum for '{filename}': {e.reason if hasattr(e, 'reason') else e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching OSC spectrum for '{filename}': {e}")
            raise RuntimeError(f"Unexpected error fetching OSC spectrum for '{filename}': {e}")
