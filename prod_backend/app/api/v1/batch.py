from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional, List, Dict, Any, Union
import json
import logging
from app.core.dependencies import (
    get_sqlalchemy_spectrum_repository,
    get_model_factory,
    get_osc_spectrum_repo,
    get_app_settings
)
from app.domain.services.spectrum_service import SpectrumService
from app.domain.services.classification_service import ClassificationService
from app.domain.services.spectrum_processing_service import SpectrumProcessingService
from app.domain.services.batch_processing_service import BatchProcessingService
from app.shared.utils.helpers import sanitize_for_json
from app.shared.schemas.classification import ClassificationSchema

logger = logging.getLogger("batch_api")
router = APIRouter()

@router.post("/batch-process")
async def batch_process(
    params: str = Form('{}'),
    zip_file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    model_id: Optional[str] = Form(None),
    db_repo = Depends(get_sqlalchemy_spectrum_repository),
    model_factory = Depends(get_model_factory),
    osc_repo = Depends(get_osc_spectrum_repo),
    settings = Depends(get_app_settings)
):
    """
    Batch classification endpoint that accepts either:
    - A zip file containing multiple spectrum files, OR
    - A list of individual spectrum files

    Supports:
    - Multiple file types: .fits, .dat, .txt, .lnw, .csv
    - Multiple model types: dash, transformer, user_uploaded
    - Processing parameters: smoothing, redshift, wavelength range, RLAP calculation

    Usage:
    - For zip file: Send zip_file parameter
    - For individual files: Send files parameter (list of files)
    """
    try:
        logger.info("/api/v1/batch-process endpoint called")
        logger.info(f"Received model_id: {model_id}")

        # Validate input - must have either zip_file OR files, not both
        if zip_file and files:
            raise HTTPException(
                status_code=400,
                detail="Cannot provide both zip_file and files. Use one or the other."
            )
        if not zip_file and not files:
            raise HTTPException(
                status_code=400,
                detail="Must provide either zip_file or files parameter."
            )

        # Parse parameters
        parsed_params = json.loads(params) if params else {}

        # Determine model type
        if model_id:
            logger.info(f"Using user-uploaded model: {model_id}")
            model_type = "user_uploaded"
        else:
            model_type = parsed_params.get('modelType', 'dash')
            if model_type not in ['dash', 'transformer']:
                model_type = 'dash'
            logger.info(f"Using model type: {model_type}")

        # Initialize services
        spectrum_service = SpectrumService(db_repo, osc_repo)
        classification_service = ClassificationService(model_factory)
        processing_service = SpectrumProcessingService()
        batch_service = BatchProcessingService(
            spectrum_service, classification_service, processing_service
        )

        # Determine input type and process
        if zip_file:
            logger.info(f"Processing zip file: {zip_file.filename}")
            input_files = zip_file
        else:
            file_count = len(files) if files else 0
            logger.info(f"Processing {file_count} individual files")
            input_files = files

        # Process the batch
        results = await batch_service.process_batch(
            input_files, parsed_params, model_type, model_id
        )

        logger.info(f"Batch processing completed successfully.")
        return sanitize_for_json(results)

    except HTTPException:
        # Re-raise HTTP exceptions (like validation errors)
        raise
    except Exception as e:
        logger.error(f"Exception in batch_process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch process error: {e}")

# Keeping old endpoint so it works when called by frontend, will remove later
@router.post("/batch-process-multiple", deprecated=True)
async def batch_process_multiple(
    params: str = Form('{}'),
    files: List[UploadFile] = File(...),
    model_id: Optional[str] = Form(None),
    db_repo = Depends(get_sqlalchemy_spectrum_repository),
    model_factory = Depends(get_model_factory),
    osc_repo = Depends(get_osc_spectrum_repo),
    settings = Depends(get_app_settings)
):
    """
    DEPRECATED: Use /batch-process with files parameter instead.

    This endpoint is kept for backward compatibility but will be removed in a future version.
    """
    logger.warning("Using deprecated /batch-process-multiple endpoint. Use /batch-process instead.")

    # Delegate
    return await batch_process(
        params=params,
        files=files,
        model_id=model_id,
        db_repo=db_repo,
        model_factory=model_factory,
        osc_repo=osc_repo,
        settings=settings
    )
