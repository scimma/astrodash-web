from typing import Optional
from domain.models.spectrum import Spectrum
from domain.models.classification import Classification
from infrastructure.ml.model_factory import ModelFactory
from config.settings import Settings, get_settings

class ClassificationService:
    def __init__(self, model_factory: ModelFactory, settings: Optional[Settings] = None):
        """Service for classification operations. Injects model factory and settings."""
        self.model_factory = model_factory
        self.settings = settings or get_settings()

    async def classify_spectrum(
        self,
        spectrum: Spectrum,
        model_type: str,
        user_model_id: Optional[str] = None
    ) -> Classification:
        classifier = self.model_factory.get_classifier(model_type, user_model_id)
        results = await classifier.classify(spectrum)
        if not results:
            raise ValueError("Classification failed or returned no results.")
        return Classification(
            spectrum_id=getattr(spectrum, 'id', None),
            model_type=model_type,
            user_model_id=user_model_id,
            results=results
        )
