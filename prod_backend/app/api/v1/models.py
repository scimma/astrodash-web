from fastapi import APIRouter, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.core.dependencies import get_model_service
from app.domain.services.model_service import ModelService
from app.shared.schemas.user_model import UserModelSchema, ModelUploadResponse, UserModelInfo, ModelInfoResponse
from app.domain.models.user_model import UserModel

from app.shared.utils.validators import ValidationError
from app.config.logging import get_logger
from app.core.exceptions import (
    ValidationException,
    ModelNotFoundException,
    ModelConflictException,
    ModelValidationException,
    ConfigurationException
)

logger = get_logger(__name__)
router = APIRouter()

@router.post("/upload-model", response_model=ModelUploadResponse)
async def upload_model(
    file: UploadFile = File(...),
    class_mapping: str = Form(...),
    input_shape: str = Form(...),
    name: str = Form(""),
    description: str = Form(""),
    owner: str = Form(""),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Upload and validate a user's PyTorch model.

    This endpoint performs:
    1. File extension validation
    2. Class mapping validation
    3. Input shape validation
    4. Model loading and compatibility checking
    5. Output shape verification
    6. Metadata extraction
    """
    try:
        # Read file content
        model_content = await file.read()

        # Upload and validate model using service
        user_model, model_info = await model_service.upload_model(
            model_content=model_content,
            filename=file.filename,
            class_mapping_str=class_mapping,
            input_shape_str=input_shape,
            name=name if name else file.filename,  # Use provided name or default to filename
            description=description,
            owner=owner
        )

        # Prepare response
        response_data = {
            "status": "success",
            "message": "Model uploaded and validated successfully.",
            "model_id": user_model.id,
            "model_filename": file.filename,
            "class_mapping": model_info.get("class_mapping"),
            "output_shape": model_info.get("output_shape"),
            "input_shape": model_info.get("input_shapes", [None])[0] if model_info.get("input_shapes") else None,
            "model_info": model_info
        }

        logger.info(f"Model uploaded successfully: {user_model.id}")
        return ModelUploadResponse(**response_data)

    except ValidationError as e:
        logger.warning(f"Model upload validation failed: {e}")
        raise ModelValidationException(str(e))
    except Exception as e:
        logger.error(f"Model upload failed: {e}")
        raise ConfigurationException(f"Internal server error: {str(e)}")

@router.get("/models/health")
async def models_health_check():
    """
    Health check for models endpoint.
    """
    return {"status": "healthy", "message": "Models endpoint is working"}

@router.get("/models", response_model=List[UserModelSchema])
async def list_models(
    model_service: ModelService = Depends(get_model_service)
):
    """
    List all uploaded models with additional metadata.
    """
    try:
        models = await model_service.list_models()

        # Load additional metadata from service
        # Note: The enhanced metadata loading should be moved to the service layer
        enhanced_models = []

        for model in models:
            try:
                # Get basic model data
                model_data = model.__dict__.copy()

                # Load additional metadata from storage
                try:
                    # Get comprehensive model info from service
                    model_info = model_service.get_model_info(model.id)
                    class_mapping = model_info.get('class_mapping', {})
                    input_shape = model_info.get('input_shape', [])
                    metadata = {k: v for k, v in model_info.items() if k not in ['class_mapping', 'input_shape']}

                    # Add the additional fields that frontend expects
                    model_data.update({
                        "class_mapping": class_mapping,
                        "input_shape": input_shape,
                        "model_filename": model.name,  # Use the name field which contains the original filename
                        "model_id": model.id  # Ensure model_id is available
                    })
                except Exception as e:
                    logger.warning(f"Failed to load metadata for model {model.id}: {e}")
                    # Provide fallback values
                    model_data.update({
                        "class_mapping": None,
                        "input_shape": None,
                        "model_filename": model.name,  # Use the name field which contains the original filename
                        "model_id": model.id
                    })

                enhanced_models.append(UserModelSchema(**model_data))

            except Exception as e:
                logger.error(f"Failed to process model {model.id}: {e}")
                # Skip this model if there's an error
                continue

        return enhanced_models

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []

@router.get("/models/{model_id}", response_model=ModelInfoResponse)
async def get_model_info(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Get detailed information about a specific model.
    """
    try:
        # Get model info from service
        model_info = model_service.get_model_info(model_id)

        return ModelInfoResponse(**model_info)

    except ValidationError as e:
        logger.warning(f"Failed to get model info for {model_id}: {e}")
        raise ModelValidationException(str(e))
    except Exception as e:
        logger.error(f"Failed to get model info for {model_id}: {e}")
        raise ConfigurationException(f"Failed to get model info: {str(e)}")

@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Delete a model and all associated files.
    """
    try:
        await model_service.delete_model(model_id)

        return JSONResponse(
            content={
                "status": "success",
                "message": f"Model {model_id} deleted successfully."
            }
        )

    except ValidationError as e:
        logger.warning(f"Failed to delete model {model_id}: {e}")
        raise ModelValidationException(str(e))
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}")
        raise ConfigurationException(f"Failed to delete model: {str(e)}")

@router.put("/models/{model_id}")
async def update_model(
    model_id: str,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Update model metadata.
    """
    try:
        updates = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description

        if not updates:
            raise ModelValidationException("No updates provided")

        updated_model = await model_service.update_model_metadata(model_id, updates)

        return JSONResponse(
            content={
                "status": "success",
                "message": f"Model {model_id} updated successfully.",
                "model_id": updated_model.id
            }
        )

    except ValidationError as e:
        logger.warning(f"Failed to update model {model_id}: {e}")
        raise ModelValidationException(str(e))
    except Exception as e:
        logger.error(f"Failed to update model {model_id}: {e}")
        raise ConfigurationException(f"Failed to update model: {str(e)}")

@router.get("/models/owner/{owner}", response_model=List[UserModelSchema])
async def list_models_by_owner(
    owner: str,
    model_service: ModelService = Depends(get_model_service)
):
    """
    List models by owner.
    """
    try:
        models = await model_service.list_models_by_owner(owner)
        return [UserModelSchema(**m.__dict__) for m in models]
    except ValidationError as e:
        logger.warning(f"Failed to list models for owner {owner}: {e}")
        raise ModelValidationException(str(e))
    except Exception as e:
        logger.error(f"Failed to list models for owner {owner}: {e}")
        raise ConfigurationException(f"Failed to list models: {str(e)}")
