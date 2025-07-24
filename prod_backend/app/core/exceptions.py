from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR
import logging

class AppException(Exception):
    """Base exception for the application."""
    def __init__(self, message: str, status_code: int = HTTP_400_BAD_REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class SpectrumNotFoundException(AppException):
    """Raised when a spectrum is not found."""
    def __init__(self, spectrum_id: str):
        super().__init__(f"Spectrum with ID '{spectrum_id}' not found.", status_code=HTTP_404_NOT_FOUND)

class ModelNotFoundException(AppException):
    """Raised when a model is not found."""
    def __init__(self, model_id: str):
        super().__init__(f"Model with ID '{model_id}' not found.", status_code=HTTP_404_NOT_FOUND)

class ClassificationException(AppException):
    """Raised for errors during classification."""
    def __init__(self, message: str = "Classification failed."):
        super().__init__(message, status_code=HTTP_400_BAD_REQUEST)

class StorageException(AppException):
    """Raised for storage-related errors."""
    def __init__(self, message: str = "Storage error."):
        super().__init__(message, status_code=HTTP_500_INTERNAL_SERVER_ERROR)

# Global exception handler registration

def register_exception_handlers(app):
    """Register global exception handlers with the FastAPI app."""
    logger = logging.getLogger("exceptions")

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.error(f"AppException: {exc.message}", exc_info=True)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message}
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled Exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."}
        )
