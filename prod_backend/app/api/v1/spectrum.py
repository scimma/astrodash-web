from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional
from shared.schemas.spectrum import SpectrumSchema
from shared.schemas.classification import ClassificationSchema
from core.dependencies import get_sqlalchemy_spectrum_repository, get_osc_spectrum_repo, get_model_factory, get_app_settings
from domain.services.spectrum_service import SpectrumService
from domain.services.classification_service import ClassificationService
from domain.models.spectrum import Spectrum
from shared.utils.validators import validate_file_extension

router = APIRouter()

@router.post("/process", response_model=ClassificationSchema)
async def process_spectrum(
    params: str = Form('{}'),
    file: Optional[UploadFile] = File(None),
    model_id: Optional[str] = Form(None),
    db_repo = Depends(get_sqlalchemy_spectrum_repository),
    osc_repo = Depends(get_osc_spectrum_repo),
    model_factory = Depends(get_model_factory),
    settings = Depends(get_app_settings)
):
    spectrum_service = SpectrumService(db_repo, osc_repo)
    classification_service = ClassificationService(model_factory)

    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Validate file extension for spectrum files
    try:
        validate_file_extension(file.filename, [".dat", ".lnw", ".txt"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file extension: {e}")

    try:
        spectrum = await spectrum_service.get_spectrum_from_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse spectrum file: {e}")

    # Save spectrum to DB (already done in get_spectrum_from_file if repo does it, else ensure here)
    # spectrum = await db_repo.save(spectrum)  # Not needed if repo already saves

    # Classify spectrum
    try:
        result = await classification_service.classify_spectrum(spectrum, model_type="dash")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    return ClassificationSchema(**result.__dict__)
