from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.core.dependencies import get_app_settings, get_sqlalchemy_model_repository
from app.domain.services.model_service import ModelService
from app.shared.schemas.user_model import UserModelSchema
from app.domain.models.user_model import UserModel
from app.infrastructure.storage.model_storage import ModelStorage
from app.shared.utils.validators import ValidationError
import logging

logger = logging.getLogger("models_api")
router = APIRouter()

class ModelUploadResponse(BaseModel):
    """Enhanced response model for model uploads."""
    status: str
    message: str
    model_id: Optional[str] = None
    model_filename: Optional[str] = None
    class_mapping: Optional[Dict[str, int]] = None
    output_shape: Optional[List[int]] = None
    input_shape: Optional[List[int]] = None
    model_info: Optional[Dict[str, Any]] = None

class UserModelInfo(BaseModel):
    model_id: str
    description: str = ""

class ModelInfoResponse(BaseModel):
    """Response model for detailed model information."""
    model_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    uploaded_at: Optional[str] = None
    file_size_bytes: Optional[int] = None
    class_mapping: Optional[Dict[str, int]] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    n_classes: Optional[int] = None
    model_type: Optional[str] = None
    total_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None

@router.post("/upload-model", response_model=ModelUploadResponse)
async def upload_model(
    file: UploadFile = File(...),
    class_mapping: str = Form(...),
    input_shape: str = Form(...),
    description: str = Form(""),
    owner: str = Form(""),
    settings = Depends(get_app_settings),
    repo = Depends(get_sqlalchemy_model_repository)
):
    """
    Upload and validate a PyTorch model with comprehensive validation.

    This endpoint performs:
    1. File extension validation
    2. Class mapping validation
    3. Input shape validation
    4. Model loading and compatibility checking
    5. Output shape verification
    6. Metadata extraction
    """
    try:
        # Initialize storage and service
        user_model_dir = settings.user_model_dir
        model_storage = ModelStorage(user_model_dir)
        service = ModelService(repo, model_storage)

        # Read file content
        model_content = await file.read()

        # Upload and validate model using service
        user_model, model_info = await service.upload_model(
            model_content=model_content,
            filename=file.filename,
            class_mapping_str=class_mapping,
            input_shape_str=input_shape,
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
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": str(e),
                "model_id": None,
                "model_filename": None,
                "class_mapping": None,
                "output_shape": None,
                "input_shape": None,
                "model_info": None
            }
        )
    except Exception as e:
        logger.error(f"Model upload failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}",
                "model_id": None,
                "model_filename": None,
                "class_mapping": None,
                "output_shape": None,
                "input_shape": None,
                "model_info": None
            }
        )

@router.get("/models/health")
async def models_health_check():
    """
    Health check for models endpoint.
    """
    return {"status": "healthy", "message": "Models endpoint is working"}

@router.get("/models", response_model=List[UserModelSchema])
async def list_models(repo = Depends(get_sqlalchemy_model_repository)):
    """
    List all uploaded models.
    """
    try:
        service = ModelService(repo)
        models = await service.list_models()
        return [UserModelSchema(**m.__dict__) for m in models]
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        # Return empty list instead of failing
        return []

@router.get("/models/{model_id}", response_model=ModelInfoResponse)
async def get_model_info(
    model_id: str,
    repo = Depends(get_sqlalchemy_model_repository),
    settings = Depends(get_app_settings)
):
    """
    Get detailed information about a specific model.
    """
    try:
        user_model_dir = settings.user_model_dir
        model_storage = ModelStorage(user_model_dir)
        service = ModelService(repo, model_storage)

        # Get model info from service
        model_info = service.get_model_info(model_id)

        return ModelInfoResponse(**model_info)

    except ValidationError as e:
        logger.warning(f"Failed to get model info for {model_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get model info for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    repo = Depends(get_sqlalchemy_model_repository),
    settings = Depends(get_app_settings)
):
    """
    Delete a model and all associated files.
    """
    try:
        user_model_dir = settings.user_model_dir
        model_storage = ModelStorage(user_model_dir)
        service = ModelService(repo, model_storage)

        await service.delete_model(model_id)

        return JSONResponse(
            content={
                "status": "success",
                "message": f"Model {model_id} deleted successfully."
            }
        )

    except ValidationError as e:
        logger.warning(f"Failed to delete model {model_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@router.put("/models/{model_id}")
async def update_model(
    model_id: str,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    repo = Depends(get_sqlalchemy_model_repository),
    settings = Depends(get_app_settings)
):
    """
    Update model metadata.
    """
    try:
        user_model_dir = settings.user_model_dir
        model_storage = ModelStorage(user_model_dir)
        service = ModelService(repo, model_storage)

        updates = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description

        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        updated_model = await service.update_model_metadata(model_id, updates)

        return JSONResponse(
            content={
                "status": "success",
                "message": f"Model {model_id} updated successfully.",
                "model_id": updated_model.id
            }
        )

    except ValidationError as e:
        logger.warning(f"Failed to update model {model_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")

@router.get("/models/owner/{owner}", response_model=List[UserModelSchema])
async def list_models_by_owner(
    owner: str,
    repo = Depends(get_sqlalchemy_model_repository)
):
    """
    List models by owner.
    """
    try:
        service = ModelService(repo)
        models = await service.list_models_by_owner(owner)
        return [UserModelSchema(**m.__dict__) for m in models]
    except ValidationError as e:
        logger.warning(f"Failed to list models for owner {owner}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to list models for owner {owner}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
