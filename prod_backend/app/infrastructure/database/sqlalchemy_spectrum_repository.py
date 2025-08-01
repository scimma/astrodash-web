from typing import Optional, Any
from sqlalchemy.orm import Session
from app.domain.repositories.spectrum_repository import SpectrumRepository
from app.domain.models.spectrum import Spectrum
from app.infrastructure.database.models import SpectrumDB
from datetime import datetime
import uuid

class SQLAlchemySpectrumRepository(SpectrumRepository):
    """
    Concrete repository for spectra using SQLAlchemy.
    Maps between Spectrum domain model and SpectrumDB ORM model.
    """
    def __init__(self, db: Session):
        self.db = db

    async def save(self, spectrum: Spectrum) -> Spectrum:
        db_spectrum = SpectrumDB(
            id=spectrum.id or str(uuid.uuid4()),
            osc_ref=spectrum.osc_ref,
            file_name=spectrum.file_name,
            x=spectrum.x,
            y=spectrum.y,
            redshift=spectrum.redshift,
            meta=spectrum.meta,
            created_at=getattr(spectrum, 'created_at', datetime.utcnow())
        )

        # Use add instead of merge for new objects, and handle the session properly
        self.db.add(db_spectrum)
        self.db.commit()

        # Only refresh if the object is in the session
        try:
            self.db.refresh(db_spectrum)
        except Exception:
            # If refresh fails, just return the domain object without refreshing
            pass

        return self._to_domain(db_spectrum)

    async def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        db_spectrum = self.db.query(SpectrumDB).filter(SpectrumDB.id == spectrum_id).first()
        return self._to_domain(db_spectrum) if db_spectrum else None

    async def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        db_spectrum = self.db.query(SpectrumDB).filter(SpectrumDB.osc_ref == osc_ref).first()
        return self._to_domain(db_spectrum) if db_spectrum else None

    async def get_from_file(self, file: Any) -> Optional[Spectrum]:
        """Read spectrum from file and save to database."""
        import pandas as pd
        import io
        import logging

        logger = logging.getLogger("sqlalchemy_spectrum_repository")
        filename = getattr(file, 'filename', 'unknown')
        logger.info(f"Reading spectrum file: {filename}")

        try:
            # Handle file reading
            file_obj = file
            if hasattr(file, 'filename') and hasattr(file, 'file'):
                # This is a FastAPI UploadFile - get the underlying file object
                file_obj = file.file

            # Handle different file types
            if filename.lower().endswith('.lnw'):
                spectrum = await self._read_lnw_file(file_obj, filename)
            elif filename.lower().endswith(('.dat', '.txt')):
                spectrum = await self._read_text_file(file_obj, filename)
            elif filename.lower().endswith('.fits'):
                spectrum = await self._read_fits_file(file_obj, filename)
            else:
                logger.error(f"Unsupported file format: {filename}")
                return None

            if spectrum:
                # Temporarily disable database saving - just return the spectrum
                # return await self.save(spectrum)
                return spectrum
            return None

        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}", exc_info=True)
            return None

    async def _read_lnw_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .lnw file with specific wavelength filtering."""
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
                return None

            wavelength, flux = zip(*spectrum)
            return Spectrum(x=list(wavelength), y=list(flux), file_name=filename)

        except Exception as e:
            import logging
            logger = logging.getLogger("sqlalchemy_spectrum_repository")
            logger.error(f"Error reading .lnw file {filename}: {e}", exc_info=True)
            return None

    async def _read_text_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .dat/.txt file."""
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

            # Parse with pandas
            try:
                df = pd.read_csv(file_stream, sep='\s+', header=None, comment='#')
            except Exception:
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
                    return None

                df = pd.DataFrame(spectrum_data, columns=[0, 1])

            if len(df.columns) >= 2:
                wavelength = df.iloc[:, 0].to_numpy()
                flux = df.iloc[:, 1].to_numpy()
                return Spectrum(x=wavelength.tolist(), y=flux.tolist(), file_name=filename)
            else:
                return None

        except Exception as e:
            import logging
            logger = logging.getLogger("sqlalchemy_spectrum_repository")
            logger.error(f"Error reading text file {filename}: {e}", exc_info=True)
            return None

    async def _read_fits_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read FITS file."""
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
                return None
            wavelength_col = next((col for col in ['wavelength', 'WAVE', 'LAMBDA'] if col in data.names), None)
            flux_col = next((col for col in ['flux', 'FLUX', 'SPEC'] if col in data.names), None)
            if not wavelength_col or not flux_col:
                return None
            wavelength = data[wavelength_col]
            flux = data[flux_col]
            return Spectrum(x=wavelength.tolist(), y=flux.tolist(), file_name=filename)
        except Exception as e:
            import logging
            logger = logging.getLogger("sqlalchemy_spectrum_repository")
            logger.error(f"Error reading FITS file {filename}: {e}")
            return None

    def _to_domain(self, db_spectrum: SpectrumDB) -> Spectrum:
        return Spectrum(
            x=db_spectrum.x,
            y=db_spectrum.y,
            redshift=db_spectrum.redshift,
            id=db_spectrum.id,
            osc_ref=db_spectrum.osc_ref,
            file_name=db_spectrum.file_name,
            meta=db_spectrum.meta
        )
