from typing import Any

class BaseClassifier:
    """
    Abstract base classifier interface. All classifiers should implement an async classify method.
    """
    def __init__(self, config=None):
        self.config = config

    async def classify(self, spectrum: Any) -> dict:
        raise NotImplementedError("Subclasses must implement classify()")
