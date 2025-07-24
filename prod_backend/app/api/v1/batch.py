from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional, List
from core.dependencies import get_sqlalchemy_spectrum_repository, get_osc_spectrum_repo, get_model_factory, get_app_settings
from domain.services.spectrum_service import SpectrumService
from domain.services.classification_service import ClassificationService
from shared.schemas.classification import ClassificationSchema
import zipfile
import io
from shared.utils.validators import validate_file_extension

router = APIRouter()

@router.post("/batch-process")
async def batch_process(
    params: str = Form('{}'),
    zip_file: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
    db_repo = Depends(get_sqlalchemy_spectrum_repository),
    model_factory = Depends(get_model_factory),
    osc_repo = Depends(get_osc_spectrum_repo),
    settings = Depends(get_app_settings)
):
    spectrum_service = SpectrumService(db_repo, osc_repo)
    classification_service = ClassificationService(model_factory)
    results = []
    try:
        contents = await zip_file.read()
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            for name in zf.namelist():
                if name.endswith(('.dat', '.lnw')):
                    with zf.open(name) as f:
                        file_bytes = f.read()
                        file_obj = UploadFile(filename=name, file=io.BytesIO(file_bytes))
                        # Validate file extension
                        try:
                            validate_file_extension(name, [".dat", ".lnw", ".txt"])
                        except Exception as e:
                            results.append({"file": name, "error": f"Invalid file extension: {e}"})
                            continue
                        try:
                            spectrum = await spectrum_service.get_spectrum_from_file(file_obj)
                            result = await classification_service.classify_spectrum(spectrum, model_type="dash")
                            results.append(ClassificationSchema(**result.__dict__))
                        except Exception as e:
                            results.append({"file": name, "error": str(e)})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process zip file: {e}")
    return {"results": results}

@router.post("/batch-process-multiple")
async def batch_process_multiple(
    params: str = Form('{}'),
    files: List[UploadFile] = File(...),
    model_id: Optional[str] = Form(None),
    db_repo = Depends(get_sqlalchemy_spectrum_repository),
    model_factory = Depends(get_model_factory),
    osc_repo = Depends(get_osc_spectrum_repo),
    settings = Depends(get_app_settings)
):
    spectrum_service = SpectrumService(db_repo, osc_repo)
    classification_service = ClassificationService(model_factory)
    results = []
    for file in files:
        # Validate file extension
        try:
            validate_file_extension(file.filename, [".dat", ".lnw", ".txt"])
        except Exception as e:
            results.append({"file": getattr(file, 'filename', 'unknown'), "error": f"Invalid file extension: {e}"})
            continue
        try:
            spectrum = await spectrum_service.get_spectrum_from_file(file)
            result = await classification_service.classify_spectrum(spectrum, model_type="dash")
            results.append(ClassificationSchema(**result.__dict__))
        except Exception as e:
            results.append({"file": getattr(file, 'filename', 'unknown'), "error": str(e)})
    return {"results": results}
