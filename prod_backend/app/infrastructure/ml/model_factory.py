from typing import Optional
from infrastructure.ml.classifiers.base import BaseClassifier
from infrastructure.ml.classifiers.dash_classifier import DashClassifier
from infrastructure.ml.classifiers.transformer_classifier import TransformerClassifier
from infrastructure.ml.classifiers.user_classifier import UserClassifier

class ModelFactory:
    """
    Factory for creating classifier instances based on model type and user model ID.
    """
    def __init__(self, config=None):
        self.config = config

    def get_classifier(
        self,
        model_type: str,
        user_model_id: Optional[str] = None
    ) -> BaseClassifier:
        if user_model_id:
            return UserClassifier(user_model_id, self.config)
        if model_type == "dash":
            return DashClassifier(self.config)
        elif model_type == "transformer":
            return TransformerClassifier(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
