from fastapi import FastAPI
from api.v1.spectrum import router as spectrum_router
from api.v1.classification import router as classification_router
from api.v1.models import router as models_router
from api.v1.batch import router as batch_router
from config.logging import init_logging
from core.exceptions import register_exception_handlers
import logging

init_logging()
logger = logging.getLogger("main")

app = FastAPI()
register_exception_handlers(app)

# Include routers under /api/v1
app.include_router(spectrum_router, prefix="/api/v1", tags=["spectrum"])
app.include_router(classification_router, prefix="/api/v1", tags=["classification"])
app.include_router(models_router, prefix="/api/v1", tags=["models"])
app.include_router(batch_router, prefix="/api/v1", tags=["batch"])

logger.info("AstroDash API server starting up.")

# Optionally, add a root health check
@app.get("/")
async def root():
    logger.info("Health check endpoint called.")
    return {"message": "AstroDash API is running"}
