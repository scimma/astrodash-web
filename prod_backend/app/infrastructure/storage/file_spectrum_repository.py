from typing import Optional, Any
from app.domain.models.spectrum import Spectrum
from app.domain.repositories.spectrum_repository import SpectrumRepository
from app.infrastructure.ml.processors.data_processor import DashSpectrumProcessor
from app.config.settings import get_settings, Settings
from app.config.logging import get_logger
import os
import json
import uuid
import httpx
import urllib3
from app.shared.utils.validators import validate_spectrum_data, validate_redshift

# Suppress SSL warnings since we're using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = get_logger(__name__)

class FileSpectrumRepository(SpectrumRepository):
    """
    File-based repository for spectra. Stores spectra as JSON files in a directory.
    Uses DashSpectrumProcessor to parse files.
    """
    def __init__(self, config: Settings = None):
        self.config = config or get_settings()
        self.processor = DashSpectrumProcessor(w0=4000, w1=9000, nw=1024)
        self.storage_dir = os.path.join(self.config.storage_dir, "spectra")
        os.makedirs(self.storage_dir, exist_ok=True)

    async def save(self, spectrum: Spectrum) -> Spectrum:
        try:
            validate_spectrum_data(spectrum.x, spectrum.y)
            if spectrum.redshift is not None:
                validate_redshift(spectrum.redshift)
            if not spectrum.id:
                spectrum.id = str(uuid.uuid4())
            path = os.path.join(self.storage_dir, f"{spectrum.id}.json")
            with open(path, "w") as f:
                json.dump({
                    "id": spectrum.id,
                    "osc_ref": spectrum.osc_ref,
                    "file_name": spectrum.file_name,
                    "x": spectrum.x,
                    "y": spectrum.y,
                    "redshift": spectrum.redshift,
                    "meta": spectrum.meta
                }, f)
            logger.info(f"Saved spectrum {spectrum.id} to {path}")
            return spectrum
        except Exception as e:
            logger.error(f"Error saving spectrum: {e}", exc_info=True)
            raise

    async def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        path = os.path.join(self.storage_dir, f"{spectrum_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return Spectrum(
            id=data["id"],
            osc_ref=data.get("osc_ref"),
            file_name=data.get("file_name"),
            x=data["x"],
            y=data["y"],
            redshift=data.get("redshift"),
            meta=data.get("meta", {})
        )

    async def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        # Not implemented for file-based repo
        return None

    async def get_from_file(self, file: Any) -> Optional[Spectrum]:
        # Accepts UploadFile or file-like object
        filename = getattr(file, 'filename', 'unknown')
        logger.info(f"Reading spectrum file: {filename}")

        try:
            import pandas as pd
            import io
            import numpy as np

            # Handle file reading like the old backend
            file_obj = file
            if hasattr(file, 'filename') and hasattr(file, 'file'):
                # This is a FastAPI UploadFile - get the underlying file object
                file_obj = file.file

            # Handle different file types like the old backend
            if filename.lower().endswith('.lnw'):
                return await self._read_lnw_file(file_obj, filename)
            elif filename.lower().endswith(('.dat', '.txt')):
                return await self._read_text_file(file_obj, filename)
            elif filename.lower().endswith('.fits'):
                return await self._read_fits_file(file_obj, filename)
            else:
                logger.error(f"Unsupported file format: {filename}")
                return None

        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}", exc_info=True)
            return None

    async def _read_lnw_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .lnw file with specific wavelength filtering like the old backend."""
        try:
            import re

            # Read file contents
            if hasattr(file_obj, 'read'):
                file_obj.seek(0)
                content = file_obj.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
            else:
                with open(file_obj, 'r') as f:
                    content = f.read()

            # Parse like the old backend
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
                    # Apply wavelength filtering like the old backend
                    if 2000 <= w <= 11000:
                        spectrum.append((w, f))
                except Exception:
                    continue

            if not spectrum:
                logger.error(f"No valid spectrum data found in .lnw file: {filename}")
                return None

            wavelength, flux = zip(*spectrum)
            logger.info(f"Read .lnw file: {filename} with {len(wavelength)} points")

            # Create and save spectrum
            spectrum_obj = Spectrum(x=list(wavelength), y=list(flux), file_name=filename)

            # Validate before saving
            if not spectrum_obj.is_valid():
                logger.error("Spectrum validation failed for .lnw file")
                return None

            saved_spectrum = await self.save(spectrum_obj)
            return saved_spectrum

        except Exception as e:
            logger.error(f"Error reading .lnw file {filename}: {e}", exc_info=True)
            return None

    async def _read_text_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .dat/.txt file like the old backend."""
        try:
            import pandas as pd
            import io

            # Read file contents
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            if hasattr(file_obj, 'read'):
                content = file_obj.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                file_stream = io.StringIO(content)
            else:
                file_stream = file_obj

            # Parse with pandas like the old backend
            try:
                df = pd.read_csv(file_stream, sep='\s+', header=None, comment='#')
            except Exception as pandas_error:
                logger.warning(f"Pandas parsing failed for {filename}, trying fallback method")

                # Fallback: manual line-by-line parsing
                file_stream.seek(0)
                lines = file_stream.readlines()
                spectrum_data = []

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            w = float(parts[0])
                            f = float(parts[1])
                            spectrum_data.append([w, f])
                        except (ValueError, IndexError):
                            continue

                if len(spectrum_data) == 0:
                    logger.error("Fallback parsing also failed - no valid data found")
                    return None

                df = pd.DataFrame(spectrum_data, columns=[0, 1])

            if len(df.columns) >= 2:
                wavelength = df.iloc[:, 0].to_numpy()
                flux = df.iloc[:, 1].to_numpy()
                logger.info(f"Read text file: {filename} with {len(wavelength)} points")

                # Create and save spectrum
                spectrum_obj = Spectrum(x=wavelength.tolist(), y=flux.tolist(), file_name=filename)

                # Validate before saving
                if not spectrum_obj.is_valid():
                    logger.error("Spectrum validation failed for text file")
                    return None

                saved_spectrum = await self.save(spectrum_obj)
                return saved_spectrum
            else:
                logger.error(f"Text file must contain at least two columns: {filename}")
                return None

        except Exception as e:
            logger.error(f"Error reading text file {filename}: {e}", exc_info=True)
            return None

    async def _read_fits_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read FITS file like the old backend."""
        try:
            from astropy.io import fits
            with fits.open(file_obj) as hdul:
                data = None
                for hdu in hdul:
                    try:
                        if hdu.data is not None and hasattr(hdu.data, 'names'):
                            if 'wavelength' in hdu.data.names or 'WAVE' in hdu.data.names:
                                data = hdu.data
                                break
                    except AttributeError:
                        continue
            if data is None:
                logger.error("No suitable data HDU found in FITS file.")
                return None
            wavelength_col = next((col for col in ['wavelength', 'WAVE', 'LAMBDA'] if col in data.names), None)
            flux_col = next((col for col in ['flux', 'FLUX', 'SPEC'] if col in data.names), None)
            if not wavelength_col or not flux_col:
                logger.error("Could not find wavelength or flux columns in FITS data.")
                return None
            wavelength = data[wavelength_col]
            flux = data[flux_col]
            logger.info(f"Read FITS file: {filename} with {len(wavelength)} points")

            # Create and save spectrum
            spectrum_obj = Spectrum(x=wavelength.tolist(), y=flux.tolist(), file_name=filename)

            # Validate before saving
            if not spectrum_obj.is_valid():
                logger.error("Spectrum validation failed for FITS file")
                return None

            saved_spectrum = await self.save(spectrum_obj)
            return saved_spectrum
        except Exception as e:
            logger.error(f"Error reading FITS file {filename}: {e}")
            return None

class OSCSpectrumRepository(SpectrumRepository):
    """
    Repository for retrieving spectra from the Open Supernova Catalog (OSC) API.
    Only get_by_osc_ref is implemented.
    """
    def __init__(self, config: Settings = None):
        self.config = config or get_settings()
        self.processor = DashSpectrumProcessor(w0=4000, w1=9000, nw=1024)
        self.osc_api_url = self.config.osc_api_url

    async def save(self, spectrum: Spectrum) -> Spectrum:
        raise NotImplementedError("Saving spectra is not supported for OSC repository.")

    async def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        raise NotImplementedError("Retrieving by ID is not supported for OSC repository.")

    async def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        try:
            logger.info(f"OSC repository: Starting to fetch OSC reference: {osc_ref}")

            # Extract the object name from the OSC reference (e.g., "osc-sn2002er-8" -> "sn2002er")
            if osc_ref.startswith('osc-'):
                obj_name = osc_ref.split('-')[1]  # Get "sn2002er" from "osc-sn2002er-8"
            else:
                obj_name = osc_ref

            logger.info(f"OSC repository: Extracted object name: {obj_name}")

            # Use the same URL structure as the old backend
            url = f"{self.osc_api_url}/{obj_name}/spectra/time+data"
            logger.info(f"OSC repository: Fetching OSC spectrum from: {url}")

            # Use httpx with SSL verification completely disabled (like the old backend)
            # Create a custom SSL context that doesn't verify certificates
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            async with httpx.AsyncClient(
                timeout=30.0,
                verify=False,
                http2=False,  # Disable HTTP/2 to avoid SSL issues
                follow_redirects=True  # Handle 301 redirects
            ) as client:
                logger.info(f"OSC repository: Making HTTP request to {url}")
                response = await client.get(url)
                logger.info(f"OSC repository: OSC API response status: {response.status_code}")

                if response.status_code != 200:
                    logger.error(f"OSC repository: OSC API returned status {response.status_code}")
                    return None

                data = response.json()
                logger.info(f"OSC repository: OSC API response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

                # Use the same parsing logic as the old backend
                if obj_name in data and "spectra" in data[obj_name]:
                    spectra = data[obj_name]["spectra"]
                    logger.info(f"OSC repository: Found {len(spectra)} spectra for {obj_name}")

                    if spectra and len(spectra) > 0:
                        # Get the first spectrum data (same as old backend)
                        spectrum_data = spectra[0][1]  # [wavelength, flux] pairs
                        logger.info(f"OSC repository: First spectrum has {len(spectrum_data)} data points")

                        # Convert to numpy arrays and transpose (same as old backend)
                        import numpy as np
                        wave, flux = np.array(spectrum_data).T.astype(float)

                        # Convert back to lists for the Spectrum model
                        x = wave.tolist()
                        y = flux.tolist()

                        # Get redshift if available, default to 0.0 (same as old backend)
                        redshift = data[obj_name].get("redshift", 0.0)

                        logger.info(f"OSC repository: Successfully parsed OSC spectrum: {len(x)} wavelength points, redshift={redshift}")

                        # Generate a unique ID for the spectrum
                        spectrum_id = f"osc-{obj_name}-{uuid.uuid4().hex[:8]}"

                        spectrum = Spectrum(
                            id=spectrum_id,
                            x=x,
                            y=y,
                            redshift=redshift,
                            osc_ref=osc_ref,
                            file_name=f"{obj_name}.json",
                            meta={"source": "osc", "object_name": obj_name}
                        )

                        logger.info(f"OSC repository: Created spectrum object: {spectrum}")
                        logger.info(f"OSC repository: Spectrum validation result: {spectrum.is_valid()}")

                        return spectrum
                else:
                    logger.error(f"OSC repository: Invalid OSC API response structure for {obj_name}")

                return None

        except httpx.ConnectError as e:
            logger.error(f"OSC repository: Network connection error for OSC API: {e}")
            return None
        except httpx.TimeoutException as e:
            logger.error(f"OSC repository: Request timeout for OSC API: {e}")
            return None
        except Exception as e:
            logger.error(f"OSC repository: Unexpected error fetching OSC spectrum: {e}", exc_info=True)
            return None

    async def get_from_file(self, file: Any) -> Optional[Spectrum]:
        return None
