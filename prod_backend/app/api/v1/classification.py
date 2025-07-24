from fastapi import APIRouter, Query, Depends
from shared.schemas.spectrum import SpectrumSchema
from core.dependencies import get_app_settings

router = APIRouter()

@router.get("/template-spectrum", response_model=SpectrumSchema)
async def get_template_spectrum(sn_type: str = Query('Ia'), age_bin: str = Query('2 to 6'), settings = Depends(get_app_settings)):
    # TODO: Retrieve template spectrum from service/config using settings
    return SpectrumSchema(x=[], y=[])

@router.get("/analysis-options")
async def get_analysis_options(settings = Depends(get_app_settings)):
    # TODO: Return options from service/config using settings
    return {"sn_types": [], "age_bins": []}
