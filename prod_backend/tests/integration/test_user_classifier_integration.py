import os
import sys
import pytest
import numpy as np
import torch

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../app"))
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from domain.models.spectrum import Spectrum
from infrastructure.ml.classifiers.user_classifier import UserClassifier

USER_MODEL_ID = "932eed3d-4d0e-4594-a490-5fd4f5e7a344"
USER_MODEL_DIR = "/home/jesusca/code_personal/astrodash-web/backend/astrodash_models/user_uploaded"

@pytest.mark.asyncio
async def test_user_classifier_integration():
    model_base = os.path.join(USER_MODEL_DIR, USER_MODEL_ID)
    model_path = model_base + '.pth'
    mapping_path = model_base + '.classes.json'
    input_shape_path = model_base + '.input_shape.json'
    if not (os.path.exists(model_path) and os.path.exists(mapping_path) and os.path.exists(input_shape_path)):
        pytest.skip("No real user-uploaded model available for integration test.")
    classifier = UserClassifier(user_model_id=USER_MODEL_ID, config={"user_model_dir": USER_MODEL_DIR})
    spectrum = Spectrum(x=[1.0]*1024, y=[2.0]*1024, redshift=0.0)
    results = await classifier.classify(spectrum)
    assert "best_matches" in results
    assert len(results["best_matches"]) > 0
