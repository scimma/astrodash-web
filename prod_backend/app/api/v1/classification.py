from fastapi import APIRouter, Query, Depends, HTTPException
from app.shared.schemas.spectrum import SpectrumSchema
from app.core.dependencies import get_app_settings
import numpy as np
import os
import logging

logger = logging.getLogger("classification_api")
router = APIRouter()

# Template spectrum endpoint moved to spectrum.py where it belongs
