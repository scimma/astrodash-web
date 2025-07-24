from typing import Optional, List
from domain.models.user_model import UserModel
from domain.repositories.model_repository import ModelRepository
from core.exceptions import ModelNotFoundException

class ModelService:
    """
    Service layer for user-uploaded model operations.
    Orchestrates business logic and repository access for user models.
    """

    def __init__(self, model_repo: ModelRepository):
        self.model_repo = model_repo

    async def save_model(self, model: UserModel) -> UserModel:
        # Example: enforce unique name or other business rules here
        return await self.model_repo.save(model)

    async def get_model(self, model_id: str) -> UserModel:
        model = await self.model_repo.get_by_id(model_id)
        if not model:
            raise ModelNotFoundException(model_id)
        return model

    async def list_models(self) -> List[UserModel]:
        return await self.model_repo.list_all()

    async def delete_model(self, model_id: str) -> None:
        model = await self.model_repo.get_by_id(model_id)
        if not model:
            raise ModelNotFoundException(model_id)
        await self.model_repo.delete(model_id)

    async def list_models_by_owner(self, owner: str) -> List[UserModel]:
        return await self.model_repo.get_by_owner(owner)
