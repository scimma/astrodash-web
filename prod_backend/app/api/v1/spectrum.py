from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional
from shared.schemas.spectrum import SpectrumSchema
from shared.schemas.classification import ClassificationSchema
from core.dependencies import get_sqlalchemy_spectrum_repository, get_osc_spectrum_repo, get_model_factory, get_app_settings
from domain.services.spectrum_service import SpectrumService
from domain.services.classification_service import ClassificationService
from domain.models.spectrum import Spectrum
from shared.utils.validators import validate_file_extension
from domain.services.redshift_service import RedshiftService
from shared.utils.helpers import prepare_log_wavelength_and_templates, get_nonzero_minmax
import json
import io

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

    # Read spectrum
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

@router.post("/estimate-redshift")
async def estimate_redshift(
    file: UploadFile = File(...),
    sn_type: str = Form(...),
    age_bin: str = Form(...),
    template_filename: str = Form('sn_and_host_templates.npz'),
    redshift_service: RedshiftService = Depends(RedshiftService),
):
    # Parse the uploaded spectrum file
    contents = await file.read()
    file_stream = io.StringIO(contents.decode("utf-8"))
    lines = [line.strip() for line in file_stream if line.strip() and not line.startswith("#")]
    x, y = [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
            except Exception:
                continue
    # Prepare templates and log-wavelength grid
    processed_data = {"x": x, "y": y}
    log_wave, input_flux_log, snTemplates, dwlog, nw, w0, w1 = prepare_log_wavelength_and_templates(processed_data, template_filename)
    # Assuming get_templates_for_type_age is defined elsewhere or will be added
    # template_fluxes, template_names, template_minmax_indexes = get_templates_for_type_age(snTemplates, sn_type, age_bin, log_wave)
    # input_minmax_index = get_nonzero_minmax(input_flux_log)
    # Estimate redshift
    # est_z, _, _, est_z_err = await redshift_service.estimate_redshift(
    #     input_flux_log, template_fluxes, nw, dwlog, input_minmax_index, template_minmax_indexes, template_names
    # )
    # return {
    #     "estimated_redshift": float(est_z) if est_z is not None else None,
    #     "estimated_redshift_error": float(est_z_err) if est_z_err is not None else None,
    #     "message": "Redshift estimated successfully" if est_z is not None else "Redshift estimation failed"
    # }
    # Placeholder for get_templates_for_type_age and get_nonzero_minmax
    # These functions are not defined in the provided context, so they are commented out.
    # If they are meant to be added, they should be defined here.
    # For now, returning a placeholder response.
    return {"estimated_redshift": None, "estimated_redshift_error": None, "message": "Redshift estimation failed (dependencies not available)"}
