import numpy as np
from scipy import signal
from astropy.io import fits
import pandas as pd
import requests
import os
import sys

class SpectrumProcessor:
    def __init__(self):
        self.supported_formats = ['.fits', '.dat', '.txt']
        self.osc_base_url = "https://api.astrocats.space/"

    def read_file(self, file_or_ref):
        """Read spectrum data from various sources"""
        if isinstance(file_or_ref, str):
            # Check if it's an OSC reference
            if file_or_ref.startswith('osc-'):
                return self._read_osc_spectrum(file_or_ref)
            else:
                raise ValueError("Invalid file reference")
        else:
            # Handle file upload
            filename = file_or_ref.filename.lower()
            if filename.endswith('.fits'):
                return self._read_fits(file_or_ref)
            elif filename.endswith(('.dat', '.txt')):
                return self._read_text(file_or_ref)
            else:
                raise ValueError(f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}")

    def _read_osc_spectrum(self, osc_ref):
        """Read spectrum from Open Supernova Catalog"""
        try:
            # Parse osc_ref: 'osc-sn2002er-8' -> event: 'sn2002er', index: 8
            if '-' not in osc_ref[4:]:
                raise ValueError("OSC reference must be in the form 'osc-<event>-<index>' (e.g., osc-sn2002er-8)")
            event_and_idx = osc_ref[4:]
            event, idx_str = event_and_idx.rsplit('-', 1)
            try:
                idx = int(idx_str)
            except ValueError:
                raise ValueError("OSC reference index must be an integer (e.g., osc-sn2002er-8)")

            # Query all spectra for the event
            url = f"{self.osc_base_url}{event}/spectra/"
            response = requests.get(url)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            if event not in data or 'spectra' not in data[event] or not data[event]['spectra']:
                print(f"OSC API response for {event}:", data, file=sys.stderr)
                raise ValueError("No spectrum data found in OSC response. See server logs for details.")

            spectra = data[event]['spectra']
            if idx < 0 or idx >= len(spectra):
                print(f"OSC API response for {event} (index {idx} out of range):", data, file=sys.stderr)
                raise ValueError(f"Spectrum index {idx} out of range for event {event}. See server logs for details.")

            spectrum = spectra[idx]
            if 'data' not in spectrum:
                print(f"OSC API response for {event} (no 'data' in spectrum at index {idx}):", spectrum, file=sys.stderr)
                raise ValueError("No data points found in spectrum. See server logs for details.")

            # Process OSC data format
            wavelength = []
            flux = []
            for point in spectrum['data']:
                if len(point) >= 2:
                    try:
                        wavelength.append(float(point[0]))
                        flux.append(float(point[1]))
                    except (ValueError, TypeError):
                        continue  # Skip invalid points

            if not wavelength or not flux:
                print(f"OSC API response for {event} (no valid points at index {idx}):", spectrum, file=sys.stderr)
                raise ValueError("No valid data points found in spectrum. See server logs for details.")

            print(f"Raw flux values from OSC for {osc_ref} (first 10): {flux[:10]}", file=sys.stderr)
            print(f"Min raw flux: {np.min(flux)}, Max raw flux: {np.max(flux)}", file=sys.stderr)

            return {'x': wavelength, 'y': flux}
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch spectrum from OSC: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading OSC spectrum: {str(e)}")

    def _read_fits(self, file):
        """Read spectrum data from FITS file"""
        try:
            with fits.open(file) as hdul:
                # Assuming the spectrum is in the first extension (often 1 or 2)
                # Try common HDU indices, or iterate
                data = None
                for hdu in hdul:
                    if hdu.data is not None and ('wavelength' in hdu.data.names or 'WAVE' in hdu.data.names):
                        data = hdu.data
                        break
                if data is None:
                    raise ValueError("No suitable data HDU found in FITS file.")

                # Try common wavelength and flux column names
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
        """Read spectrum data from text file"""
        try:
            # Try to read as CSV/TSV, allowing various delimiters
            df = pd.read_csv(file, sep=None, engine='python', header=None)
            if len(df.columns) >= 2:
                # Convert to numpy arrays for consistent processing
                wavelength = df.iloc[:, 0].to_numpy()
                flux = df.iloc[:, 1].to_numpy()
                return {'x': wavelength.tolist(), 'y': flux.tolist()}
            else:
                raise ValueError("Text file must contain at least two columns")
        except Exception as e:
            raise ValueError(f"Error reading text file: {str(e)}")

    def process(self, x, y, smoothing, known_z, z_value, min_wave, max_wave, classify_host, calculate_rlap):
        """Process spectrum data with given parameters"""
        wavelength = np.array(x)
        flux = np.array(y)

        # Apply wavelength range limits
        if min_wave is not None and max_wave is not None:
            mask = (wavelength >= min_wave) & (wavelength <= max_wave)
            wavelength = wavelength[mask]
            flux = flux[mask]
            if len(wavelength) == 0:
                raise ValueError("No spectrum data within the specified wavelength range.")

        # Normalize flux to a 0-1 range (subtract min and divide by range)
        if len(flux) > 0:
            min_flux_orig = np.min(flux)
            max_flux_orig = np.max(flux)
            flux_range = max_flux_orig - min_flux_orig

            if flux_range > 0: # Avoid division by zero
                flux = (flux - min_flux_orig) / flux_range
                print(f"Flux normalized to 0-1 range. Original min: {min_flux_orig:.2e}, max: {max_flux_orig:.2e}", file=sys.stderr)
            elif flux_range == 0 and min_flux_orig != 0: # All values are the same non-zero value
                flux = np.zeros_like(flux) # Normalize to 0 if all values are identical and non-zero
                print(f"Flux normalized to 0 (all original values were identical and non-zero: {min_flux_orig:.2e}).", file=sys.stderr)
            else: # All values are zero
                print("Flux values are all zero, no normalization applied.", file=sys.stderr)
        else:
            print("No flux data to normalize.", file=sys.stderr)

        # Apply smoothing if requested (smoothing > 0)
        if smoothing > 0 and len(flux) > smoothing * 2:
            print(f"Applying smoothing (level: {smoothing})...", file=sys.stderr)
            print(f"Flux before smoothing - Min: {np.min(flux):.4f}, Max: {np.max(flux):.4f}", file=sys.stderr)
            # savgol_filter requires window_length to be odd and polyorder <= window_length
            window_length = smoothing * 2 + 1
            if window_length > len(flux):
                window_length = len(flux) if len(flux) % 2 == 1 else len(flux) - 1
            if window_length < 3: # Minimum window length is 3
                window_length = 3

            polyorder = 3
            if polyorder >= window_length:
                polyorder = window_length - 1

            print(f"Savgol filter params: window_length={window_length}, polyorder={polyorder}", file=sys.stderr)
            flux = signal.savgol_filter(flux, window_length, polyorder)
            print(f"Flux after smoothing - Min: {np.min(flux):.4f}, Max: {np.max(flux):.4f}", file=sys.stderr)

        # Placeholder for redshift calculation if not known
        if not known_z:
            # In a real scenario, you'd calculate redshift here based on the spectrum
            calculated_z = 0.05 # Mock calculated redshift for now
            redshift = calculated_z
        else:
            redshift = float(z_value) if z_value is not None else None

        # Placeholder for host classification
        host_classification = "No Host" # Mock host classification
        if classify_host:
            # In a real scenario, you'd classify host here
            pass

        # Placeholder for RLAP calculation
        rlap_score = None
        if calculate_rlap:
            # In a real scenario, you'd calculate rlap here
            rlap_score = 7.5 # Mock rlap score

        return {
            'x': wavelength.tolist(),
            'y': flux.tolist(),
            'redshift': redshift,
            'host_classification': host_classification,
            'rlap_score': rlap_score
        }

    def normalize_spectrum(self, wavelength, flux):
        """Normalize spectrum data"""
        # Remove continuum
        continuum = np.polyfit(wavelength, flux, 3)
        continuum_flux = np.polyval(continuum, wavelength)
        normalized_flux = flux - continuum_flux

        # Normalize to unit area
        area = np.trapz(normalized_flux, wavelength)
        if area != 0:
            normalized_flux = normalized_flux / area

        return normalized_flux

    def calculate_redshift(self, wavelength, flux, template_wavelength, template_flux):
        """Calculate redshift using cross-correlation"""
        # This is a placeholder for the actual redshift calculation
        return 0.0
