from fastapi import APIRouter
from app.api.v1.spectrum import router as spectrum_router
from app.api.v1.classification import router as classification_router
from app.api.v1.models import router as models_router
from app.api.v1.batch import router as batch_router

# Main API router for v1 endpoints
router = APIRouter()

# Include all individual routers
router.include_router(spectrum_router, tags=["spectrum"])
router.include_router(classification_router, tags=["classification"])
router.include_router(models_router, tags=["models"])
router.include_router(batch_router, tags=["batch"])
