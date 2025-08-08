from fastapi import FastAPI
from app.api.v1.router import router as api_v1_router
from app.config.logging import init_logging, get_logger
from app.core.exceptions import register_exception_handlers
from app.core.middleware import setup_middleware
from app.core.monitoring import get_health_status

# Initialize logging
init_logging()
logger = get_logger(__name__)

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

# Include API v1 router
app.include_router(api_v1_router, prefix="/api/v1")

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
