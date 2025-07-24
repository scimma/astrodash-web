from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class UserModelSchema(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the user model")
    name: Optional[str] = Field(None, description="Name of the user model")
    description: Optional[str] = Field(None, description="Description of the user model")
    owner: Optional[str] = Field(None, description="Owner of the user model")
    model_path: Optional[str] = Field(None, description="Path to the model weights file")
    class_mapping_path: Optional[str] = Field(None, description="Path to the class mapping file")
    input_shape_path: Optional[str] = Field(None, description="Path to the input shape file")
    created_at: Optional[datetime] = Field(None, description="Datetime when the model was created")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "id": "932eed3d-4d0e-4594-a490-5fd4f5e7a344",
                "name": "My Custom SN Classifier",
                "description": "A user-uploaded model for supernova classification.",
                "owner": "user123",
                "model_path": "app/astrodash_models/user_uploaded/932eed3d-4d0e-4594-a490-5fd4f5e7a344.pth",
                "class_mapping_path": "app/astrodash_models/user_uploaded/932eed3d-4d0e-4594-a490-5fd4f5e7a344.classes.json",
                "input_shape_path": "app/astrodash_models/user_uploaded/932eed3d-4d0e-4594-a490-5fd4f5e7a344.input_shape.json",
                "created_at": "2024-06-01T12:00:00Z",
                "meta": {"framework": "PyTorch", "num_classes": 5}
            }
        }
