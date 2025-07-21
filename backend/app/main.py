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
import torch
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid

from .services.spectrum_processor import SpectrumProcessor
from .services.ml_classifier import MLClassifier, TransformerClassifier
from .services.astrodash_backend import get_training_parameters, AgeBinning, load_template_spectrum, get_valid_sn_types_and_age_bins
from .services.redshift_estimator import get_median_redshift
from .services.utils import sanitize_for_json, normalize_age_bin
from app.services.rlap_calculator import calculate_rlap_with_redshift

logging.basicConfig(
    level=logging.INFO, # CHANGE BACK TO INFO, DEBUR, ERROR
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
    allow_origins=["http://localhost:3000", "http://localhost:4000"],  # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service Instantiation
spectrum_processor = SpectrumProcessor()

# Change user model directory to astrodash_models/user_uploaded
ASTRODASH_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'astrodash_models')
USER_MODEL_DIR = os.path.join(ASTRODASH_MODELS_DIR, 'user_uploaded')
os.makedirs(USER_MODEL_DIR, exist_ok=True)

class ModelUploadResponse(BaseModel):
    status: str
    message: str
    model_id: str | None = None
    model_filename: str | None = None
    class_mapping: dict | None = None
    output_shape: list | None = None
    input_shape: list | None = None

@app.post("/api/upload-model", response_model=ModelUploadResponse)
async def upload_model(
    file: UploadFile = File(...),
    class_mapping: str = Form(...),
    input_shape: str = Form(...)
):
    # 1. Check file extension
    if not (file.filename and (file.filename.endswith('.pth') or file.filename.endswith('.pt'))):
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": "Only .pth or .pt files are allowed."
        })
    # 2. Parse class mapping
    try:
        class_map = json.loads(class_mapping)
        if not isinstance(class_map, dict) or not class_map:
            raise ValueError
    except Exception:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": "Invalid class mapping. Must be a non-empty JSON object."
        })
    # 3. Parse input shape
    try:
        input_shape_list = json.loads(input_shape) if isinstance(input_shape, str) else input_shape
        if (not isinstance(input_shape_list, list) or
            not all(isinstance(x, int) and x > 0 for x in input_shape_list)):
            raise ValueError
    except Exception:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": "Invalid input shape. Must be a JSON list of positive integers, e.g., [1, 1, 32, 32]."
        })
    # 4. Assign a unique model_id
    model_id = str(uuid.uuid4())
    model_base = os.path.join(USER_MODEL_DIR, model_id)
    model_path = model_base + '.pth'
    # 5. Save model file (temporarily)
    content = await file.read()
    with open(model_path, 'wb') as f:
        f.write(content)
    # 6. Try to load model and check output shape
    try:
        import torch
        # Load as TorchScript
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        # Prepare dummy inputs matching the model's forward signature
        # Assume input_shape_list is [1, 1024] for each input
        dummy_wavelength = torch.randn(*input_shape_list)
        dummy_flux = torch.randn(*input_shape_list)
        dummy_redshift = torch.randn(1, 1)
        with torch.no_grad():
            output = model(dummy_wavelength, dummy_flux, dummy_redshift)
        output_shape = list(output.shape)
        n_classes = len(class_map)
        if output_shape[-1] != n_classes:
            os.remove(model_path)
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": f"Model output shape {output_shape} does not match number of classes {n_classes}."
            })
    except Exception as e:
        if os.path.exists(model_path):
            os.remove(model_path)
        logger.error(f"Model upload failed: {e}")
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": f"Failed to load model: {str(e)}"
        })
    # 7. Save class mapping and input shape only if validation passed
    mapping_path = model_base + '.classes.json'
    with open(mapping_path, 'w') as f:
        json.dump(class_map, f)
    input_shape_path = model_base + '.input_shape.json'
    with open(input_shape_path, 'w') as f:
        json.dump(input_shape_list, f)
    return {
        "status": "success",
        "message": "Model uploaded and validated successfully.",
        "model_id": model_id,
        "model_filename": os.path.basename(model_path),
        "class_mapping": class_map,
        "output_shape": output_shape,
        "input_shape": input_shape_list
    }

def get_classifier(model_type: str):
    """Get the appropriate classifier based on model type"""
    if model_type == 'transformer':
        logger.info("Instantiating TransformerClassifier")
        return TransformerClassifier()
    elif model_type == 'user_uploaded':
        logger.info("User-uploaded model - this should be handled separately")
        raise ValueError("User-uploaded models should be handled in the main process function")
    else:
        logger.info("Instantiating MLClassifier (Dash)")
        return MLClassifier()

@app.get("/")
async def read_root():
    return {"message": "Welcome to Astrodash API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/osc-references", summary="Get available OSC references")
async def get_osc_references():
    """
    Get a list of available OSC references for testing.
    """
    try:
        logger.info("Requested OSC references")
        # Return a list of OSC references for testing
        references = [
            "https://www.wiserep.org/spectrum/view/12345",
            "https://www.wiserep.org/spectrum/view/67890",
            "https://www.wiserep.org/spectrum/view/11111",
            "https://www.wiserep.org/spectrum/view/22222",
            "https://www.wiserep.org/spectrum/view/33333"
        ]
        return {
            "status": "success",
            "references": references,
            "message": "OSC references retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error fetching OSC references: {e}")
        return {
            "status": "error",
            "message": f"Failed to fetch OSC references: {str(e)}"
        }

@app.post("/process")
async def process_spectrum(
    params: str = Form('{}'),
    file: Optional[UploadFile] = File(None),
    model_id: Optional[str] = Form(None)
):
    try:
        logger.info("/process endpoint called")
        logger.info(f"Received params: {params}")
        logger.info(f"Received file: {file.filename if file else 'None'}")
        logger.info(f"Received model_id: {model_id}")
        parsed_params = json.loads(params) if params else {}
        logger.info(f"Parsed params: {parsed_params}")

        # Check if model_id is provided first (user-uploaded model)
        if model_id:
            logger.info(f"Using user-uploaded model: {model_id}")
            model_type = "user_uploaded"  # Set model_type for user-uploaded models
        else:
            # Extract model type from parameters, default to 'dash'
            model_type = parsed_params.get('modelType', 'dash')
            if model_type not in ['dash', 'transformer']:
                model_type = 'dash'  # Default to dash if invalid model type
            logger.info(f"Using model type: {model_type}")
        # Handle spectrum data - either from file or OSC reference
        spectrum_data = None
        try:
            if file:
                logger.info(f"Processing uploaded file: {file.filename}")
                spectrum_data = spectrum_processor.read_file(file)
            elif 'oscRef' in parsed_params:
                osc_ref = parsed_params.get('oscRef')
                logger.info(f"Processing OSC reference: {osc_ref}")
                if osc_ref:
                    spectrum_data = spectrum_processor.read_file(osc_ref)
        except RuntimeError as e:
            logger.error(f"RuntimeError in spectrum reading: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Exception in spectrum reading: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Spectrum reading error: {e}")
        if not spectrum_data:
            logger.error("No spectrum file or OSC reference provided")
            raise HTTPException(status_code=400, detail="No spectrum file or OSC reference provided")
        logger.info("About to process spectrum...")
        # Process spectrum
        try:
            logger.info(f"Spectrum data loaded, shape: {len(spectrum_data.get('x', []))} points")
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
        logger.info("About to classify spectrum...")
        # If model_id is provided, use the user-uploaded model
        if model_id:
            logger.info(f"Using user-uploaded model: {model_id}")
            model_base = os.path.join(USER_MODEL_DIR, model_id)
            model_path = model_base + '.pth'
            mapping_path = model_base + '.classes.json'
            input_shape_path = model_base + '.input_shape.json'
            # Load class mapping and input shape
            try:
                with open(mapping_path, 'r') as f:
                    class_map = json.load(f)
                with open(input_shape_path, 'r') as f:
                    input_shape_list = json.load(f)
                import torch
                # Load as TorchScript
                model = torch.jit.load(model_path, map_location='cpu')
                model.eval()
                                # Prepare input for the model
                flux = np.array(processed_data['y'])
                wavelength = np.array(processed_data['x'])
                redshift = processed_data.get('redshift', 0.0)

                logger.info(f"User model input shape: {input_shape_list}")
                logger.info(f"Flux shape: {flux.shape}, Wavelength shape: {wavelength.shape}")

                # Handle different input shapes based on model type
                if len(input_shape_list) == 4:  # [batch, channels, height, width] - CNN style
                    flat_size = np.prod(input_shape_list[1:])
                    flux_flat = np.zeros(flat_size)
                    n = min(len(flux), flat_size)
                    flux_flat[:n] = flux[:n]
                    model_input = torch.tensor(flux_flat, dtype=torch.float32).reshape(input_shape_list)
                    with torch.no_grad():
                        output = model(model_input)
                else:  # Transformer style - needs wavelength, flux, redshift
                    # Interpolate to 1024 points if needed (like transformer model)
                    if len(flux) != 1024:
                        x_old = np.linspace(0, 1, len(flux))
                        x_new = np.linspace(0, 1, 1024)
                        flux = np.interp(x_new, x_old, flux)
                        wavelength = np.interp(x_new, x_old, wavelength)

                    wavelength_tensor = torch.tensor(wavelength, dtype=torch.float32).unsqueeze(0)  # [1, 1024]
                    flux_tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)              # [1, 1024]
                    redshift_tensor = torch.tensor([[redshift]], dtype=torch.float32)               # [1, 1]

                    with torch.no_grad():
                        output = model(wavelength_tensor, flux_tensor, redshift_tensor)
                    probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
                idx_to_label = {v: k for k, v in class_map.items()}
                top_indices = np.argsort(probs)[::-1][:3]
                matches = []
                # Check if user wants RLAP calculation (note: user-uploaded models don't support RLAP calculation)
                calculate_rlap = processed_data.get('calculate_rlap', False)
                if calculate_rlap:
                    logger.info("RLAP calculation requested but not supported by user-uploaded models. Setting RLAP to None.")

                for idx in top_indices:
                    class_name = idx_to_label.get(idx, f'unknown_class_{idx}')
                    matches.append({
                        'type': class_name,
                        'age': 'N/A',  # User models don't classify age
                        'probability': float(probs[idx]),
                        'redshift': processed_data.get('redshift', 0.0),
                        'rlap': None,  # Not calculated for user-uploaded models (RLAP requires template matching)
                        'reliable': bool(probs[idx] > 0.5)
                    })
                best_match = matches[0] if matches else {'type': 'Unknown', 'age': 'N/A', 'probability': 0.0}
                return sanitize_for_json({
                    "spectrum": processed_data,
                    "classification": {
                        "best_matches": matches,
                        "best_match": best_match,
                        "reliable_matches": best_match.get('reliable', False) if best_match else False
                    },
                    "model_type": "user_uploaded",
                    "model_id": model_id
                })
            except Exception as e:
                logger.error(f"Error using user-uploaded model: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"User model error: {e}")
        # Otherwise, use the default logic for dash/transformer models
        if model_type != "user_uploaded":
            try:
                classifier = get_classifier(model_type)
                classification_results = classifier.classify(processed_data)
                logger.info(f"Classification completed successfully with {model_type} model.")
            except Exception as e:
                logger.error(f"Exception in classification: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Classification error: {e}")
            logger.info("/process completed successfully")
            return sanitize_for_json({
                "spectrum": processed_data,
                "classification": classification_results,
                "model_type": model_type
            })
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.critical(f"Unhandled exception in /process: {e}\n{traceback.format_exc()}", exc_info=True)
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
    zip_file: UploadFile = File(...),
    model_id: Optional[str] = Form(None)
):
    """
    Accepts a .zip file containing multiple spectrum files, processes and classifies each, and returns results per file.
    """
    try:
        logger.info("/api/batch-process endpoint called")
        logger.info(f"Received model_id: {model_id}")
        parsed_params = json.loads(params) if params else {}

        # Check if model_id is provided first (user-uploaded model)
        if model_id:
            logger.info(f"Using user-uploaded model: {model_id}")
            model_type = "user_uploaded"  # Set model_type for user-uploaded models
        else:
            # Extract model type from parameters, default to 'dash'
            model_type = parsed_params.get('modelType', 'dash')
            if model_type not in ['dash', 'transformer']:
                model_type = 'dash'  # Default to dash if invalid model type
            logger.info(f"Using model type: {model_type}")

        # Get the appropriate classifier (only for non-user-uploaded models)
        if model_type != "user_uploaded":
            classifier = get_classifier(model_type)

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
                            file_like = io.BytesIO(file_obj.read())
                        else:
                            content = file_obj.read()
                            try:
                                text = content.decode('utf-8')
                                file_like = io.StringIO(text)
                            except UnicodeDecodeError:
                                file_like = io.BytesIO(content)
                        # Set .name attribute for file type detection
                        file_like.name = fname
                        # Process spectrum
                        try:
                            spectrum_data = spectrum_processor.read_file(file_like)
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
                            # Use user-uploaded model if model_type is "user_uploaded"
                            if model_type == "user_uploaded":
                                logger.info(f"Using user-uploaded model: {model_id} for batch item {fname}")
                                model_base = os.path.join(USER_MODEL_DIR, model_id)
                                model_path = model_base + '.pth'
                                mapping_path = model_base + '.classes.json'
                                input_shape_path = model_base + '.input_shape.json'
                                with open(mapping_path, 'r') as f:
                                    class_map = json.load(f)
                                with open(input_shape_path, 'r') as f:
                                    input_shape_list = json.load(f)
                                import torch
                                model = torch.jit.load(model_path, map_location='cpu')
                                model.eval()
                                flux = np.array(processed_data['y'])
                                wavelength = np.array(processed_data['x'])
                                redshift = processed_data.get('redshift', 0.0)

                                logger.info(f"User model input shape: {input_shape_list}")
                                logger.info(f"Flux shape: {flux.shape}, Wavelength shape: {wavelength.shape}")

                                # Handle different input shapes based on model type
                                if len(input_shape_list) == 4:  # [batch, channels, height, width] - CNN style
                                    flat_size = np.prod(input_shape_list[1:])
                                    flux_flat = np.zeros(flat_size)
                                    n = min(len(flux), flat_size)
                                    flux_flat[:n] = flux[:n]
                                    model_input = torch.tensor(flux_flat, dtype=torch.float32).reshape(input_shape_list)
                                    with torch.no_grad():
                                        output = model(model_input)
                                else:  # Transformer style - needs wavelength, flux, redshift
                                    # Interpolate to 1024 points if needed (like transformer model)
                                    if len(flux) != 1024:
                                        x_old = np.linspace(0, 1, len(flux))
                                        x_new = np.linspace(0, 1, 1024)
                                        flux = np.interp(x_new, x_old, flux)
                                        wavelength = np.interp(x_new, x_old, wavelength)

                                    wavelength_tensor = torch.tensor(wavelength, dtype=torch.float32).unsqueeze(0)  # [1, 1024]
                                    flux_tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)              # [1, 1024]
                                    redshift_tensor = torch.tensor([[redshift]], dtype=torch.float32)               # [1, 1]

                                    with torch.no_grad():
                                        output = model(wavelength_tensor, flux_tensor, redshift_tensor)

                                probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
                                idx_to_label = {v: k for k, v in class_map.items()}
                                # Check if user wants RLAP calculation (note: user-uploaded models don't support RLAP calculation)
                                calculate_rlap = processed_data.get('calculate_rlap', False)
                                if calculate_rlap:
                                    logger.info("RLAP calculation requested but not supported by user-uploaded models. Setting RLAP to None.")

                                top_indices = np.argsort(probs)[::-1][:3]
                                matches = []
                                for idx in top_indices:
                                    class_name = idx_to_label.get(idx, f'unknown_class_{idx}')
                                    matches.append({
                                        'type': class_name,
                                        'age': 'N/A',  # User models don't classify age
                                        'probability': float(probs[idx]),
                                        'redshift': float(processed_data.get('redshift', 0.0)),
                                        'rlap': None,  # Not calculated for user-uploaded models (RLAP requires template matching)
                                        'reliable': bool(probs[idx] > 0.5)
                                    })
                                best_match = matches[0] if matches else {'type': 'Unknown', 'age': 'N/A', 'probability': 0.0}
                                results[fname] = {
                                    "spectrum": processed_data,
                                    "classification": {
                                        "best_matches": matches,
                                        "best_match": best_match,
                                        "reliable_matches": best_match.get('reliable', False) if best_match else False
                                    },
                                    "model_type": "user_uploaded",
                                    "model_id": model_id
                                }
                                continue
                            # Use standard classifier for non-user-uploaded models
                            if model_type != "user_uploaded":
                                classification_results = classifier.classify(processed_data)
                                results[fname] = {
                                    "spectrum": processed_data,
                                    "classification": classification_results,
                                    "model_type": model_type
                                }
                        except Exception as e:
                            results[fname] = {"error": str(e)}
                except Exception as e:
                    results[fname] = {"error": str(e)}
        return to_python_type(results)
    except Exception as e:
        logger.error(f"Exception in batch_process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch process error: {e}")

@app.post("/api/batch-process-multiple")
async def batch_process_multiple(
    params: str = Form('{}'),
    files: List[UploadFile] = File(...),
    model_id: Optional[str] = Form(None)
):
    """
    Accepts multiple individual spectrum files, processes and classifies each, and returns results per file.
    """
    try:
        logger.info("/api/batch-process-multiple endpoint called")
        logger.info(f"Received model_id: {model_id}")
        parsed_params = json.loads(params) if params else {}

        # Check if model_id is provided first (user-uploaded model)
        if model_id:
            logger.info(f"Using user-uploaded model: {model_id}")
            model_type = "user_uploaded"  # Set model_type for user-uploaded models
        else:
            # Extract model type from parameters, default to 'dash'
            model_type = parsed_params.get('modelType', 'dash')
            if model_type not in ['dash', 'transformer']:
                model_type = 'dash'  # Default to dash if invalid model type
            logger.info(f"Using model type: {model_type}")

        # Get the appropriate classifier (only for non-user-uploaded models)
        if model_type != "user_uploaded":
            classifier = get_classifier(model_type)

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
                # Use user-uploaded model if model_type is "user_uploaded"
                if model_type == "user_uploaded":
                    logger.info(f"Using user-uploaded model: {model_id} for batch item {file.filename}")
                    model_base = os.path.join(USER_MODEL_DIR, model_id)
                    model_path = model_base + '.pth'
                    mapping_path = model_base + '.classes.json'
                    input_shape_path = model_base + '.input_shape.json'
                    with open(mapping_path, 'r') as f:
                        class_map = json.load(f)
                    with open(input_shape_path, 'r') as f:
                        input_shape_list = json.load(f)
                    import torch
                    model = torch.jit.load(model_path, map_location='cpu')
                    model.eval()
                    flux = np.array(processed_data['y'])
                    wavelength = np.array(processed_data['x'])
                    redshift = processed_data.get('redshift', 0.0)

                    logger.info(f"User model input shape: {input_shape_list}")
                    logger.info(f"Flux shape: {flux.shape}, Wavelength shape: {wavelength.shape}")

                    # Handle different input shapes based on model type
                    if len(input_shape_list) == 4:  # [batch, channels, height, width] - CNN style
                        flat_size = np.prod(input_shape_list[1:])
                        flux_flat = np.zeros(flat_size)
                        n = min(len(flux), flat_size)
                        flux_flat[:n] = flux[:n]
                        model_input = torch.tensor(flux_flat, dtype=torch.float32).reshape(input_shape_list)
                        with torch.no_grad():
                            output = model(model_input)
                    else:  # Transformer style - needs wavelength, flux, redshift
                        # Interpolate to 1024 points if needed (like transformer model)
                        if len(flux) != 1024:
                            x_old = np.linspace(0, 1, len(flux))
                            x_new = np.linspace(0, 1, 1024)
                            flux = np.interp(x_new, x_old, flux)
                            wavelength = np.interp(x_new, x_old, wavelength)

                        wavelength_tensor = torch.tensor(wavelength, dtype=torch.float32).unsqueeze(0)  # [1, 1024]
                        flux_tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)              # [1, 1024]
                        redshift_tensor = torch.tensor([[redshift]], dtype=torch.float32)               # [1, 1]

                        with torch.no_grad():
                            output = model(wavelength_tensor, flux_tensor, redshift_tensor)

                    probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
                    # Check if user wants RLAP calculation (note: user-uploaded models don't support RLAP calculation)
                    calculate_rlap = processed_data.get('calculate_rlap', False)
                    if calculate_rlap:
                        logger.info("RLAP calculation requested but not supported by user-uploaded models. Setting RLAP to None.")

                    idx_to_label = {v: k for k, v in class_map.items()}
                    top_indices = np.argsort(probs)[::-1][:3]
                    matches = []
                    for idx in top_indices:
                        class_name = idx_to_label.get(idx, f'unknown_class_{idx}')
                        matches.append({
                            'type': class_name,
                            'age': 'N/A',  # User models don't classify age
                            'probability': float(probs[idx]),
                            'redshift': float(processed_data.get('redshift', 0.0)),
                            'rlap': None,  # Not calculated for user-uploaded models (RLAP requires template matching)
                            'reliable': bool(probs[idx] > 0.5)
                        })
                    best_match = matches[0] if matches else {'type': 'Unknown', 'age': 'N/A', 'probability': 0.0}
                    results[file.filename] = {
                        "spectrum": processed_data,
                        "classification": {
                            "best_matches": matches,
                            "best_match": best_match,
                            "reliable_matches": best_match.get('reliable', False) if best_match else False
                        },
                        "model_type": "user_uploaded",
                        "model_id": model_id
                    }
                    continue
                # Use standard classifier for non-user-uploaded models
                if model_type != "user_uploaded":
                    classification_results = classifier.classify(processed_data)
                    results[file.filename] = {
                        "spectrum": processed_data,
                        "classification": classification_results,
                        "model_type": model_type
                    }
            except Exception as e:
                results[file.filename] = {"error": str(e)}

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

@app.get("/api/list-models")
async def list_models():
    """List all user-uploaded models with their metadata."""
    user_model_dir = os.path.join(os.path.dirname(__file__), '..', 'astrodash_models', 'user_uploaded')
    if not os.path.exists(user_model_dir):
        return {"models": []}
    models = []
    for fname in os.listdir(user_model_dir):
        if fname.endswith('.pth'):
            model_id = fname[:-4]
            model_filename = fname
            mapping_path = os.path.join(user_model_dir, model_id + '.classes.json')
            input_shape_path = os.path.join(user_model_dir, model_id + '.input_shape.json')
            try:
                with open(mapping_path, 'r') as f:
                    class_mapping = json.load(f)
                with open(input_shape_path, 'r') as f:
                    input_shape = json.load(f)
                models.append({
                    "model_id": model_id,
                    "model_filename": model_filename,
                    "class_mapping": class_mapping,
                    "input_shape": input_shape
                })
            except Exception as e:
                # Skip models with missing or invalid metadata
                continue
    return {"models": models}

def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_python_type(v) for v in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
