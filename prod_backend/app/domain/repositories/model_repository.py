from abc import ABC, abstractmethod
from typing import Optional, List
from app.domain.models.user_model import UserModel

class ModelRepository(ABC):
    """
    Abstract repository interface for user-uploaded models.
    Follows the repository pattern for decoupling domain and infrastructure.
    """

    @abstractmethod
    async def save(self, model: UserModel) -> UserModel:
        """Save a user model to persistent storage."""
        pass

    @abstractmethod
    async def get_by_id(self, model_id: str) -> Optional[UserModel]:
        """Retrieve a user model by its unique ID."""
        pass

    @abstractmethod
    async def list_all(self) -> List[UserModel]:
        """List all user models in storage."""
        pass

    @abstractmethod
    async def delete(self, model_id: str) -> None:
        """Delete a user model by its unique ID."""
        pass

    @abstractmethod
    async def get_by_owner(self, owner: str) -> List[UserModel]:
        """List all user models owned by a specific user."""
        pass
