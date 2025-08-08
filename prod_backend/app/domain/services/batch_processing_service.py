import zipfile
import io
from typing import List, Dict, Any, Optional, Union
from fastapi import UploadFile
from app.domain.services.spectrum_service import SpectrumService
from app.domain.services.classification_service import ClassificationService
from app.domain.services.spectrum_processing_service import SpectrumProcessingService
from app.domain.models.spectrum import Spectrum
from app.config.logging import get_logger
from app.core.exceptions import BatchProcessingException, ValidationException

logger = get_logger(__name__)

class BatchProcessingService:
    """
    Service for handling batch processing of spectrum files.
    Supports both zip files and individual file lists.
    """

    def __init__(
        self,
        spectrum_service: SpectrumService,
        classification_service: ClassificationService,
        processing_service: SpectrumProcessingService
    ):
        self.spectrum_service = spectrum_service
        self.classification_service = classification_service
        self.processing_service = processing_service
        self.supported_extensions = (".fits", ".dat", ".txt", ".lnw", ".csv")

    async def process_batch(
        self,
        files: Union[Any, List[Any]],  # Using Any to handle starlette.datastructures.UploadFile
        params: Dict[str, Any],
        model_type: str,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of spectrum files.

        Args:
            files: Either a zip file (UploadFile) or a list of individual files
            params: Processing parameters
            model_type: Type of model to use ('dash', 'transformer', 'user_uploaded')
            model_id: User model ID if using user_uploaded model

        Returns:
            Dictionary with results for each file

        Raises:
            BatchProcessingException: If batch processing fails
            ValidationException: If input validation fails
        """
        try:
            # Validate input
            if files is None:
                raise ValidationException("No files provided for batch processing")

            # Check if it's a zip file (UploadFile with .zip extension or single file)
            if hasattr(files, 'filename') and hasattr(files, 'read'):
                # Handle zip file or single file
                logger.info(f"Processing file: {files.filename}")
                return await self._process_zip_file(files, params, model_type, model_id)
            elif isinstance(files, list):
                # Handle list of individual files
                return await self._process_file_list(files, params, model_type, model_id)
            else:
                raise ValidationException(f"Invalid files type: {type(files)}. Expected UploadFile or List[UploadFile]")

        except (ValidationException, BatchProcessingException):
            raise
        except Exception as e:
            logger.error(f"Error in batch processing: {e}", exc_info=True)
            raise BatchProcessingException(f"Batch processing failed: {str(e)}")

    async def _process_zip_file(
        self,
        zip_file: Any,  # Using Any to handle starlette.datastructures.UploadFile
        params: Dict[str, Any],
        model_type: str,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process files from a zip archive."""
        logger.info(f"Processing zip file: {zip_file.filename}")

        results = {}
        contents = await zip_file.read()

        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            for fname in zf.namelist():
                info = zf.getinfo(fname)
                if info.is_dir():
                    continue  # Skip directories

                # Check file type support
                if not fname.lower().endswith(self.supported_extensions):
                    results[fname] = {"error": "Unsupported file type"}
                    continue

                try:
                    with zf.open(fname) as file_obj:
                        # Prepare file-like object for spectrum service
                        file_like = self._prepare_file_object(fname, file_obj)

                        # Process the file
                        result = await self._process_single_file(
                            file_like, fname, params, model_type, model_id
                        )
                        results[fname] = result

                except Exception as e:
                    logger.error(f"Error reading file {fname}: {e}")
                    results[fname] = {"error": str(e)}

        logger.info(f"Zip processing completed. Processed {len(results)} files.")
        return results

    async def _process_file_list(
        self,
        files: List[Any],  # Using Any to handle starlette.datastructures.UploadFile
        params: Dict[str, Any],
        model_type: str,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a list of individual files."""
        # Ensure files is a list and handle edge cases
        if not files:
            logger.warning("No files provided for processing")
            return {}

        if not isinstance(files, list):
            logger.error(f"Expected list of files, got {type(files)}")
            return {"error": "Invalid file list format"}

        logger.info(f"Processing {len(files)} individual files")

        results = {}

        for file in files:
            filename = getattr(file, 'filename', 'unknown')

            # Check file type support
            if not filename.lower().endswith(self.supported_extensions):
                results[filename] = {"error": "Unsupported file type"}
                continue

            try:
                # Process the file
                result = await self._process_single_file(
                    file, filename, params, model_type, model_id
                )
                results[filename] = result

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                results[filename] = {"error": str(e)}

        logger.info(f"File list processing completed. Processed {len(results)} files.")
        return results

    def _prepare_file_object(self, fname: str, file_obj) -> Any:  # Using Any to handle starlette.datastructures.UploadFile
        """Prepare a file-like object for the spectrum service."""
        ext = fname.lower().split('.')[-1]

        if ext == 'fits':
            # For FITS files, we need to read the content once
            content = file_obj.read()
            return UploadFile(filename=fname, file=io.BytesIO(content))
        else:
            # For text files, read content once and create StringIO
            content = file_obj.read()
            try:
                text = content.decode('utf-8')
                return UploadFile(filename=fname, file=io.StringIO(text))
            except UnicodeDecodeError:
                return UploadFile(filename=fname, file=io.BytesIO(content))

    async def _process_single_file(
        self,
        file: Any,  # Using Any to handle starlette.datastructures.UploadFile
        filename: str,
        params: Dict[str, Any],
        model_type: str,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a single spectrum file."""
        try:
            # Get spectrum from file
            spectrum = await self.spectrum_service.get_spectrum_from_file(file)

            # Apply processing parameters
            processed_spectrum = await self.processing_service.process_spectrum_with_params(
                spectrum, params
            )

            # Classify with appropriate model
            if model_type == "user_uploaded":
                result = await self.classification_service.classify_spectrum(
                    processed_spectrum,
                    model_type="user_uploaded",
                    user_model_id=model_id
                )
            else:
                result = await self.classification_service.classify_spectrum(
                    processed_spectrum,
                    model_type=model_type
                )

            # Format result
            return {
                "spectrum": {
                    "x": processed_spectrum.x,
                    "y": processed_spectrum.y,
                    "redshift": getattr(processed_spectrum, 'redshift', None)
                },
                "classification": result.results,
                "model_type": model_type,
                "model_id": model_id if model_type == "user_uploaded" else None
            }

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise
