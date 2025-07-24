from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
from core.dependencies import get_app_settings, get_sqlalchemy_model_repository
from domain.services.model_service import ModelService
from shared.schemas.user_model import UserModelSchema
from domain.models.user_model import UserModel
from shared.utils.validators import validate_user_model, validate_file_extension
import uuid
import os
import shutil
import json

router = APIRouter()

class ModelUploadResponse(BaseModel):
    model_id: str
    message: str

class UserModelInfo(BaseModel):
    model_id: str
    description: str = ""

@router.post("/upload-model", response_model=ModelUploadResponse)
async def upload_model(
    file: UploadFile = File(...),
    class_mapping: str = Form(...),
    input_shape: str = Form(...),
    settings = Depends(get_app_settings),
    repo = Depends(get_sqlalchemy_model_repository)
):
    service = ModelService(repo)
    model_id = str(uuid.uuid4())
    user_model_dir = settings.user_model_dir
    os.makedirs(user_model_dir, exist_ok=True)
    # Validate file extension before saving
    try:
        validate_file_extension(file.filename, [".pth", ".pt"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file extension: {e}")
    # Save files
    model_path = os.path.join(user_model_dir, model_id + ".pth")
    class_mapping_path = os.path.join(user_model_dir, model_id + ".classes.json")
    input_shape_path = os.path.join(user_model_dir, model_id + ".input_shape.json")
    with open(model_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    with open(class_mapping_path, "w") as f:
        f.write(class_mapping)
    with open(input_shape_path, "w") as f:
        f.write(input_shape)
    # Validate the model by loading and running dummy input
    try:
        input_shape_list = json.loads(input_shape) if isinstance(input_shape, str) else input_shape
        validate_user_model(model_path, input_shape_list, [".pth", ".pt"])
    except Exception as e:
        # Clean up files if validation fails
        os.remove(model_path)
        os.remove(class_mapping_path)
        os.remove(input_shape_path)
        raise HTTPException(status_code=400, detail=f"Model validation failed: {e}")
    user_model = UserModel(
        id=model_id,
        model_path=model_path,
        class_mapping_path=class_mapping_path,
        input_shape_path=input_shape_path
    )
    await service.save_model(user_model)
    return ModelUploadResponse(model_id=model_id, message="Model uploaded successfully.")

@router.get("/models", response_model=List[UserModelSchema])
async def list_models(repo = Depends(get_sqlalchemy_model_repository)):
    service = ModelService(repo)
    models = await service.list_models()
    return [UserModelSchema(**m.__dict__) for m in models]

@router.delete("/models/{model_id}")
async def delete_model(model_id: str, repo = Depends(get_sqlalchemy_model_repository)):
    service = ModelService(repo)
    await service.delete_model(model_id)
    return JSONResponse(content={"message": f"Model {model_id} deleted."})
