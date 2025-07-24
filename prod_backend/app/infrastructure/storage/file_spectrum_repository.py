from typing import Optional, Any
from domain.models.spectrum import Spectrum
from domain.repositories.spectrum_repository import SpectrumRepository
from infrastructure.ml.processors.data_processor import SpectrumProcessor
from config.settings import get_settings, Settings
import os
import json
import uuid
import httpx
from shared.utils.validators import validate_spectrum_data, validate_redshift

class FileSpectrumRepository(SpectrumRepository):
    """
    File-based repository for spectra. Stores spectra as JSON files in a directory.
    Uses SpectrumProcessor to parse files.
    """
    def __init__(self, config: Settings = None):
        self.config = config or get_settings()
        self.processor = SpectrumProcessor()
        self.storage_dir = os.path.join(self.config.storage_dir, "spectra")
        os.makedirs(self.storage_dir, exist_ok=True)

    async def save(self, spectrum: Spectrum) -> Spectrum:
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
        return spectrum

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
        try:
            if hasattr(file, "read"):
                contents = await file.read() if hasattr(file, "__aenter__") else file.read()
                import io
                file_stream = io.StringIO(contents.decode("utf-8"))
            elif isinstance(file, str):
                file_stream = open(file, "r")
            else:
                return None
            lines = [line.strip() for line in file_stream if line.strip() and not line.startswith("#")]
            x, y = [], []
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x.append(float(parts[0]))
                        y.append(float(parts[1]))
                    except Exception:
                        continue
            validate_spectrum_data(x, y)
            spectrum = Spectrum(x=x, y=y, file_name=getattr(file, 'filename', None))
            if spectrum.redshift is not None:
                validate_redshift(spectrum.redshift)
            return await self.save(spectrum)
        except Exception:
            return None

class OSCSpectrumRepository(SpectrumRepository):
    """
    Repository for retrieving spectra from the Open Supernova Catalog (OSC) API.
    Only get_by_osc_ref is implemented.
    """
    def __init__(self, config: Settings = None):
        self.config = config or get_settings()
        self.processor = SpectrumProcessor()
        self.osc_api_url = self.config.osc_api_url

    async def save(self, spectrum: Spectrum) -> Spectrum:
        raise NotImplementedError("Saving spectra is not supported for OSC repository.")

    async def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        raise NotImplementedError("Retrieving by ID is not supported for OSC repository.")

    async def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        url = f"{self.osc_api_url}/api/spectra/{osc_ref}/"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                return None
            data = response.json()
            # Parse the OSC API response to extract x, y, redshift, etc.
            x = data.get("wavelengths") or data.get("x")
            y = data.get("fluxes") or data.get("y")
            redshift = data.get("redshift")
            file_name = data.get("filename") or osc_ref
            meta = {k: v for k, v in data.items() if k not in ("wavelengths", "x", "fluxes", "y", "redshift", "filename")}
            if not x or not y or len(x) != len(y):
                return None
            return Spectrum(x=x, y=y, redshift=redshift, osc_ref=osc_ref, file_name=file_name, meta=meta)

    async def get_from_file(self, file: Any) -> Optional[Spectrum]:
        return None
