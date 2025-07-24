from typing import Optional, Any
from sqlalchemy.orm import Session
from domain.repositories.spectrum_repository import SpectrumRepository
from domain.models.spectrum import Spectrum
from infrastructure.database.models import SpectrumDB
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
        self.db.merge(db_spectrum)
        self.db.commit()
        self.db.refresh(db_spectrum)
        return self._to_domain(db_spectrum)

    async def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        db_spectrum = self.db.query(SpectrumDB).filter(SpectrumDB.id == spectrum_id).first()
        return self._to_domain(db_spectrum) if db_spectrum else None

    async def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        db_spectrum = self.db.query(SpectrumDB).filter(SpectrumDB.osc_ref == osc_ref).first()
        return self._to_domain(db_spectrum) if db_spectrum else None

    async def get_from_file(self, file: Any) -> Optional[Spectrum]:
        # In a real implementation, parse the file and save as a new Spectrum
        # Here, just a placeholder: expects file to be a dict with x, y, etc.
        if not isinstance(file, dict):
            return None
        spectrum = Spectrum(
            x=file.get('x'),
            y=file.get('y'),
            redshift=file.get('redshift'),
            file_name=file.get('file_name'),
            meta=file.get('meta', {})
        )
        return await self.save(spectrum)

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
