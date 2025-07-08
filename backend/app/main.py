from fastapi import FastAPI, HTTPException, UploadFile, Request, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json
import numpy as np
import os
from urllib.parse import unquote_plus
import logging
import zipfile
import io
import math

from .services.spectrum_processor import SpectrumProcessor
from .services.ml_classifier import MLClassifier
from .services.astrodash_backend import get_training_parameters, AgeBinning, load_template_spectrum, get_valid_sn_types_and_age_bins
from .services.redshift_estimator import get_median_redshift
from .services.utils import sanitize_for_json
from app.services.rlap_calculator import calculate_rlap_with_redshift

logging.basicConfig(
    level=logging.DEBUG, # CHANGE BACK TO INFO, DEBUR, ERROR
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backend.log"), # logs to a file
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Astrodash API",
    description="API for processing and classifying supernova spectra.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service Instantiation
spectrum_processor = SpectrumProcessor()
ml_classifier = MLClassifier()


@app.get("/")
async def read_root():
    """Root endpoint."""
    return {
        "message": "Welcome to Astrodash API",
        "endpoints": {
            "health": "/health",
            "process": "/process",
            "osc-references": "/api/osc-references",
            "analysis-options": "/api/analysis-options",
            "template-spectrum": "/api/template-spectrum",
            "batch-process": "api/batch-process",
            "batch-process-multiple": "api/batch-process-multiple",
            "estimate-redshift": "/api/estimate-redshift",
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/api/osc-references", summary="Get available OSC references")
async def get_osc_references():
    """
    Get a list of available Open Supernova Catalog (OSC) references.
    These are used as a source for supernova spectra data.

    TODO:
    Get actual list from wis-tns.org
    """
    try:
        available_refs = [
            'osc-sn2002er-0',
            'osc-sn2011fe-0',
            'osc-sn2014J-0',
            'osc-sn1998aq-0',
            'osc-sn1999ee-0',
            'osc-sn2005cf-0',
            'osc-sn2007af-0',
            'osc-sn2009ig-0',
            'osc-sn2012cg-0',
            'osc-sn2013dy-0',
            'osc-sn2014dt-0',
            'osc-sn2016coj-0',
        ]
        logger.info(f"Returning {len(available_refs)} OSC references.")
        return {
            'status': 'success',
            'references': available_refs
        }
    except Exception as e:
        logger.error(f"Error fetching OSC references: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_spectrum(
    params: str = Form('{}'),
    file: Optional[UploadFile] = File(None)
):
    # Get spectrum data
    try:
        logger.info("/process endpoint called")
        parsed_params = json.loads(params) if params else {}
        logger.debug(f"Parsed params: {parsed_params}")
        spectrum_data = None
        try:
            if file:
                logger.info(f"File uploaded: {file.filename}")
                spectrum_data = spectrum_processor.read_file(file)
            elif 'oscRef' in parsed_params:
                osc_ref = parsed_params.get('oscRef')
                logger.info(f"OSC reference: {osc_ref}")
                if osc_ref:
                    spectrum_data = spectrum_processor.read_file(osc_ref)
        except RuntimeError as e:
            logger.error(f"RuntimeError in spectrum reading: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Exception in spectrum reading: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Spectrum reading error: {e}")

        if not spectrum_data:
            logger.warning("No spectrum file or OSC reference provided.")
            raise HTTPException(status_code=400, detail="No spectrum file or OSC reference provided")

        # Preprocess spectrum
        try:
            processed_data = spectrum_processor.process(
                x=spectrum_data['x'],
                y=spectrum_data['y'],
                smoothing=parsed_params.get('smoothing', 0),
                known_z=parsed_params.get('knownZ', False),
                z_value=parsed_params.get('zValue'),
                min_wave=parsed_params.get('minWave'),
                max_wave=parsed_params.get('maxWave'),
                calculate_rlap=parsed_params.get('calculateRlap', False)
            )
            logger.info("Spectrum processed successfully.")
        except Exception as e:
            logger.error(f"Exception in preprocessing: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

        # Classify spectrum
        try:
            classification_results = ml_classifier.classify(processed_data)
            logger.info("Classification completed successfully.")
        except Exception as e:
            logger.error(f"Exception in classification: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Classification error: {e}")

        logger.info("/process completed successfully")
        return {
            "spectrum": processed_data,
            "classification": classification_results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"Unhandled exception in /process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis-options")
async def get_analysis_options():
    try:
        logger.info("Requested analysis options")
        backend_root = os.path.join(os.path.dirname(__file__), '..')
        npz_path = os.path.join(backend_root, 'astrodash_models', 'sn_and_host_templates.npz')
        valid = get_valid_sn_types_and_age_bins(npz_path)
        logger.info(f"Analysis options returned {repr(valid)} SN types by and age bins")
        return {
            'sn_types': list(valid.keys()),
            'age_bins_by_type': valid
        }
    except Exception as e:
        logger.error("Fetching analysis options failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/template-spectrum")
async def get_template_spectrum(sn_type: str = 'Ia', age_bin: str = '2 to 6'):
    try:
        sn_type_decoded = unquote_plus(sn_type)
        age_bin_decoded = unquote_plus(age_bin)
        logger.info(f"Requested sn_type: {repr(sn_type_decoded)}")
        logger.info(f"Requested age_bin: {repr(age_bin_decoded)}")
        pars = get_training_parameters()
        backend_root = os.path.join(os.path.dirname(__file__), '..')
        npz_path = os.path.join(backend_root, 'astrodash_models', 'sn_and_host_templates.npz')
        wave, flux = load_template_spectrum(sn_type_decoded, age_bin_decoded, npz_path, pars)
        logger.info(f"Template spectrum loaded for {sn_type_decoded} / {age_bin_decoded}")

        # Return template spectrum as-is (no preprocessing)
        return {
            'wave': list(wave),
            'flux': list(flux),
            'sn_type': sn_type_decoded,
            'age_bin': age_bin_decoded,
        }
    except FileNotFoundError:
        logger.error("Template data file not found.")
        raise HTTPException(status_code=404, detail="Template data file not found.")
    except Exception as e:
        logger.error(f"Exception in get_template_spectrum: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/line-list")
async def get_line_list():
    """Return the element/ion line list for plotting vertical lines."""
    try:
        backend_root = os.path.join(os.path.dirname(__file__), '..')
        line_list_path = os.path.join(backend_root, 'astrodash_models', 'sneLineList.txt')
        line_dict = {}
        raw_lines = []
        with open(line_list_path, 'r') as f:
            for line in f:
                raw_lines.append(line.rstrip())
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, values = line.split(':', 1)
                # Remove trailing commas and whitespace, split by comma
                wavelengths = [float(w.strip()) for w in values.replace(',', ' ').split() if w.strip()]
                line_dict[key.strip()] = wavelengths
        logger.info(f"Raw sneLineList.txt lines: {raw_lines}")
        logger.info(f"Parsed line_dict: {line_dict}")
        return line_dict
    except Exception as e:
        logger.error(f"Error reading line list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading line list: {e}")

@app.post("/api/batch-process")
async def batch_process(
    params: str = Form('{}'),
    zip_file: UploadFile = File(...)
):
    """
    Accepts a .zip file containing multiple spectrum files, processes and classifies each, and returns results per file.
    """
    try:
        logger.info("/api/batch-process endpoint called")
        parsed_params = json.loads(params) if params else {}
        results = {}
        # Read the uploaded zip file into memory
        zip_bytes = await zip_file.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for fname in z.namelist():
                info = z.getinfo(fname)
                if info.is_dir():
                    continue  # Skip directories
                # Only process supported file types
                if not fname.lower().endswith((".fits", ".dat", ".txt", ".lnw")):
                    results[fname] = {"error": "Unsupported file type"}
                    continue
                try:
                    with z.open(fname) as file_obj:
                        # Prepare file-like object for SpectrumProcessor
                        ext = fname.lower().split('.')[-1]
                        if ext == 'fits':
                            # FITS files: keep as BytesIO
                            file_like = io.BytesIO(file_obj.read())
                        else:
                            # Text files: decode bytes to string, wrap in StringIO
                            content = file_obj.read()
                            try:
                                text = content.decode('utf-8')
                            except UnicodeDecodeError:
                                text = content.decode('latin1')
                            import io as _io
                            file_like = _io.StringIO(text)
                        # Emulate UploadFile interface for SpectrumProcessor
                        class FakeUploadFile:
                            def __init__(self, filename, file):
                                self.filename = filename
                                self.file = file
                        fake_file = FakeUploadFile(fname, file_like)
                        spectrum_data = spectrum_processor.read_file(fake_file)
                        processed_data = spectrum_processor.process(
                            x=spectrum_data['x'],
                            y=spectrum_data['y'],
                            smoothing=parsed_params.get('smoothing', 0),
                            known_z=parsed_params.get('knownZ', False),
                            z_value=parsed_params.get('zValue'),
                            min_wave=parsed_params.get('minWave'),
                            max_wave=parsed_params.get('maxWave'),
                            calculate_rlap=parsed_params.get('calculateRlap', False)
                        )
                        classification_results = ml_classifier.classify(processed_data)
                        results[fname] = {
                            "spectrum": processed_data,
                            "classification": classification_results
                        }
                except Exception as e:
                    # Log first few lines of file for debugging if error occurs
                    logger.error(f"Error processing {fname}: {e}", exc_info=True)
                    results[fname] = {"error": str(e)}
        logger.info("/api/batch-process completed")
        return sanitize_for_json(results)
    except Exception as e:
        logger.critical(f"Unhandled exception in /api/batch-process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-process-multiple")
async def batch_process_multiple(
    params: str = Form('{}'),
    files: List[UploadFile] = File(...)
):
    """
    Accepts multiple individual spectrum files, processes and classifies each, and returns results per file.
    """
    try:
        logger.info("/api/batch-process-multiple endpoint called")
        parsed_params = json.loads(params) if params else {}
        results = {}

        for file in files:
            fname = file.filename
            if not fname.lower().endswith((".fits", ".dat", ".txt", ".lnw", ".csv")):
                results[fname] = {"error": "Unsupported file type"}
                continue
            try:
                spectrum_data = spectrum_processor.read_file(file)
                processed_data = spectrum_processor.process(
                    x=spectrum_data['x'],
                    y=spectrum_data['y'],
                    smoothing=parsed_params.get('smoothing', 0),
                    known_z=parsed_params.get('knownZ', False),
                    z_value=parsed_params.get('zValue'),
                    min_wave=parsed_params.get('minWave'),
                    max_wave=parsed_params.get('maxWave'),
                    calculate_rlap=parsed_params.get('calculateRlap', False)
                )
                classification_results = ml_classifier.classify(processed_data)
                results[fname] = {
                    "spectrum": processed_data,
                    "classification": classification_results
                }
            except Exception as e:
                logger.error(f"Error processing {fname}: {e}", exc_info=True)
                results[fname] = {"error": str(e)}

        logger.info("/api/batch-process-multiple completed")
        return sanitize_for_json(results)
    except Exception as e:
        logger.critical(f"Unhandled exception in /api/batch-process-multiple: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/estimate-redshift")
async def estimate_redshift(request: Request):
    data = await request.json()
    x = np.array(data.get("x"))
    y = np.array(data.get("y"))
    sn_type = data.get("type")
    age = data.get("age")
    # Load templates and parameters
    from app.services.astrodash_backend import get_training_parameters
    import os
    pars = get_training_parameters()
    w0, w1, nw = pars['w0'], pars['w1'], pars['nw']
    dwlog = np.log(w1 / w0) / nw
    log_wave = w0 * np.exp(np.arange(nw) * dwlog)
    backend_root = os.path.join(os.path.dirname(__file__), '..')
    template_path = os.path.join(backend_root, 'astrodash_models', 'sn_and_host_templates.npz')
    import numpy as np
    data_npz = np.load(template_path, allow_pickle=True)
    snTemplates_raw = data_npz['snTemplates'].item()
    snTemplates = {str(k): v for k, v in snTemplates_raw.items()}
    # Normalize age bin
    def normalize_age_bin(age):
        import re
        age = age.replace('–', '-').replace('to', '-').replace('TO', '-').replace('To', '-')
        age = age.replace(' ', '')
        match = re.match(r'(-?\d+)-(-?\d+)', age)
        if match:
            return f"{int(match.group(1))} to {int(match.group(2))}"
        return age
    age_norm = normalize_age_bin(age)
    # Find templates for this type/age
    template_fluxes = []
    template_names = []
    template_minmax_indexes = []
    if sn_type in snTemplates:
        age_bin_keys = [str(k).strip() for k in snTemplates[sn_type].keys()]
        if age_norm.strip() in age_bin_keys:
            real_key = [k for k in snTemplates[sn_type].keys() if str(k).strip() == age_norm.strip()][0]
            snInfo = snTemplates[sn_type][real_key].get('snInfo', None)
            if isinstance(snInfo, np.ndarray) and snInfo.shape[0] > 0:
                for i in range(snInfo.shape[0]):
                    template_wave = snInfo[i][0]
                    template_flux = snInfo[i][1]
                    interp_flux = np.interp(log_wave, template_wave, template_flux, left=0, right=0)
                    nonzero = np.where(interp_flux != 0)[0]
                    if len(nonzero) > 0:
                        tmin, tmax = nonzero[0], nonzero[-1]
                    else:
                        tmin, tmax = 0, len(interp_flux) - 1
                    template_fluxes.append(interp_flux)
                    template_names.append(f"{sn_type}:{age_norm}")
                    template_minmax_indexes.append((tmin, tmax))
    if not template_fluxes:
        return {"estimated_redshift": None, "estimated_redshift_error": None, "message": "No valid templates found for this type/age."}
    # Interpolate input spectrum to log-wavelength grid
    input_flux_log = np.interp(log_wave, x, y, left=0, right=0)
    input_minmax_index = (np.where(input_flux_log != 0)[0][0], np.where(input_flux_log != 0)[0][-1]) if np.any(input_flux_log != 0) else (0, len(input_flux_log) - 1)
    est_z, _, _, est_z_err = get_median_redshift(
        input_flux_log, template_fluxes, nw, dwlog, input_minmax_index, template_minmax_indexes, template_names, outerVal=0.5
    )
    return {
        "estimated_redshift": float(est_z) if est_z is not None else None,
        "estimated_redshift_error": float(est_z_err) if est_z_err is not None else None
    }
