from fastapi import FastAPI
from app.api.v1.spectrum import router as spectrum_router
from app.api.v1.classification import router as classification_router
from app.api.v1.models import router as models_router
from app.api.v1.batch import router as batch_router
from app.config.logging import init_logging
from app.core.exceptions import register_exception_handlers
from app.core.middleware import setup_middleware
from app.core.monitoring import get_health_status
import logging

# Initialize logging
init_logging()
logger = logging.getLogger("main")

# Create FastAPI app
app = FastAPI(
    title="AstroDash API",
    description="Production-grade API for astronomical spectrum classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup middleware
setup_middleware(app)

# Register exception handlers
register_exception_handlers(app)

# Include routers
app.include_router(spectrum_router, prefix="/api/v1")
app.include_router(classification_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(batch_router, prefix="/api/v1")

@app.get("/")
async def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to AstroDash API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return get_health_status()

if __name__ == "__main__":
    import uvicorn
    logger.info("AstroDash API server starting up.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
