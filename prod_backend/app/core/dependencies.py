from functools import lru_cache
from fastapi import Depends
from app.config.settings import Settings, get_settings
import logging
from typing import Generator
from sqlalchemy.orm import Session
from app.infrastructure.database.session import get_db
from app.infrastructure.database.sqlalchemy_spectrum_repository import SQLAlchemySpectrumRepository
from app.infrastructure.storage.file_spectrum_repository import OSCSpectrumRepository
from app.infrastructure.ml.model_factory import ModelFactory
from app.domain.services.template_analysis_service import TemplateAnalysisService
from app.domain.repositories.spectrum_repository import create_spectrum_template_handler

# Settings dependency (singleton)
@lru_cache()
def get_app_settings() -> Settings:
    """Get the application settings (singleton)."""
    return get_settings()

# Logger dependency
def get_logger(name: str = "app") -> logging.Logger:
    """Get a logger instance for a given name."""
    return logging.getLogger(name)

# Example: Service/repository dependencies
from app.infrastructure.storage.file_storage import FileStorage
from app.infrastructure.storage.file_spectrum_repository import FileSpectrumRepository
from app.infrastructure.ml.model_factory import ModelFactory
from app.infrastructure.database.session import get_db
from app.infrastructure.database.sqlalchemy_model_repository import SQLAlchemyModelRepository
from app.infrastructure.database.sqlalchemy_spectrum_repository import SQLAlchemySpectrumRepository

# File storage dependency
def get_file_storage(settings: Settings = Depends(get_app_settings)) -> FileStorage:
    return FileStorage(config=settings)

# Spectrum repository dependencies (file-based)
def get_file_spectrum_repo(settings: Settings = Depends(get_app_settings)) -> FileSpectrumRepository:
    return FileSpectrumRepository(config=settings)

def get_osc_spectrum_repo(settings: Settings = Depends(get_app_settings)) -> OSCSpectrumRepository:
    return OSCSpectrumRepository(config=settings)

# Model factory dependency
def get_model_factory(settings: Settings = Depends(get_app_settings)) -> ModelFactory:
    return ModelFactory(config=settings)

# SQLAlchemy user model repository dependency
def get_sqlalchemy_model_repository(db=Depends(get_db)) -> SQLAlchemyModelRepository:
    return SQLAlchemyModelRepository(db)

# SQLAlchemy spectrum repository dependency
def get_sqlalchemy_spectrum_repository(db: Session = Depends(get_db)) -> SQLAlchemySpectrumRepository:
    """Dependency to get SQLAlchemy spectrum repository."""
    return SQLAlchemySpectrumRepository(db)

def get_template_analysis_service() -> TemplateAnalysisService:
    """Dependency to get template analysis service."""
    # Create template handler for DASH model (which has templates)
    template_handler = create_spectrum_template_handler('dash')
    return TemplateAnalysisService(template_handler)
