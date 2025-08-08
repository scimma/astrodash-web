from functools import lru_cache
from fastapi import Depends
from typing import Generator
from sqlalchemy.orm import Session

# Config and logging imports
from app.config.settings import Settings, get_settings

# Database imports
from app.infrastructure.database.session import get_db
from app.infrastructure.database.sqlalchemy_spectrum_repository import SQLAlchemySpectrumRepository
from app.infrastructure.database.sqlalchemy_model_repository import SQLAlchemyModelRepository

# Storage imports
from app.infrastructure.storage.file_spectrum_repository import FileSpectrumRepository, OSCSpectrumRepository

# ML imports
from app.infrastructure.ml.model_factory import ModelFactory

# Domain services imports
from app.domain.services.template_analysis_service import TemplateAnalysisService
from app.domain.services.line_list_service import LineListService
from app.domain.services.spectrum_processing_service import SpectrumProcessingService

# Repository imports
from app.domain.repositories.spectrum_repository import create_spectrum_template_handler

# Settings dependency (singleton)
@lru_cache()
def get_app_settings() -> Settings:
    """Get the application settings (singleton)."""
    return get_settings()

# Spectrum repository dependencies (file-based)
def get_file_spectrum_repo(settings: Settings = Depends(get_app_settings)) -> FileSpectrumRepository:
    """Dependency to get file-based spectrum repository."""
    return FileSpectrumRepository(config=settings)

def get_osc_spectrum_repo(settings: Settings = Depends(get_app_settings)) -> OSCSpectrumRepository:
    """Dependency to get OSC spectrum repository."""
    return OSCSpectrumRepository(config=settings)

# Model factory dependency
def get_model_factory(settings: Settings = Depends(get_app_settings)) -> ModelFactory:
    """Dependency to get model factory."""
    return ModelFactory(config=settings)

# SQLAlchemy repository dependencies
def get_sqlalchemy_model_repository(db: Session = Depends(get_db)) -> SQLAlchemyModelRepository:
    """Dependency to get SQLAlchemy model repository."""
    return SQLAlchemyModelRepository(db)

def get_sqlalchemy_spectrum_repository(db: Session = Depends(get_db)) -> SQLAlchemySpectrumRepository:
    """Dependency to get SQLAlchemy spectrum repository."""
    return SQLAlchemySpectrumRepository(db)

# Service dependencies
def get_template_analysis_service() -> TemplateAnalysisService:
    """Dependency to get template analysis service."""
    # Create template handler for DASH model (which has templates)
    template_handler = create_spectrum_template_handler('dash')
    return TemplateAnalysisService(template_handler)

def get_line_list_service() -> LineListService:
    """Dependency to get line list service."""
    return LineListService()

def get_spectrum_processing_service() -> SpectrumProcessingService:
    """Dependency to get spectrum processing service."""
    return SpectrumProcessingService()
