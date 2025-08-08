from fastapi import APIRouter, Query, Depends, HTTPException
from app.shared.schemas.spectrum import SpectrumSchema
from app.core.dependencies import get_app_settings
from app.config.logging import get_logger
import numpy as np
import os

logger = get_logger(__name__)
router = APIRouter()

# Template spectrum endpoint moved to spectrum.py where it belongs
