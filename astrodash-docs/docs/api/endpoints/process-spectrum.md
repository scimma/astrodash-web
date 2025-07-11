---
sidebar_position: 2
---

# Process Spectrum

Process and classify a single spectrum file or OSC reference.

## Endpoint

```
POST /process
```

## Description

This is the main endpoint for processing and classifying supernova spectra. It accepts either a file upload or an OSC reference, processes the spectrum according to specified parameters, and returns both the processed spectrum data and classification results.

## Request

### Content-Type

```
multipart/form-data
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | No* | Spectrum file to upload (FITS, DAT, TXT, or LNW format) |
| `params` | String (JSON) | No | Processing parameters as JSON string |

*Either `file` or `oscRef` in params is required.

### Processing Parameters

The `params` parameter accepts a JSON string with the following fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `oscRef` | String | - | OSC reference name (e.g., "osc-sn2011fe-0") |
| `smoothing` | Integer | 0 | Smoothing parameter (0-10) |
| `knownZ` | Boolean | false | Whether redshift is known |
| `zValue` | Float | - | Redshift value (required if knownZ=true) |
| `minWave` | Float | - | Minimum wavelength in Angstroms |
| `maxWave` | Float | - | Maximum wavelength in Angstroms |
| `calculateRlap` | Boolean | false | Whether to calculate RLAP values |

## Response

### Success Response

**Status Code:** `200 OK`

```json
{
  "spectrum": {
    "x": [3500.0, 3501.0, ...],
    "y": [0.1, 0.2, ...],
    "processed": true,
    "smoothing": 6,
    "z_value": 0.5,
    "min_wave": 3500.0,
    "max_wave": 10000.0
  },
  "classification": {
    "top_match": "Ia-norm",
    "confidence": 0.95,
    "all_matches": [
      {
        "type": "Ia-norm",
        "confidence": 0.95,
        "age_bin": "4 to 8"
      },
      {
        "type": "Ia-91T",
        "confidence": 0.03,
        "age_bin": "2 to 6"
      }
    ],
    "rlap_values": {
      "Ia-norm": 0.85,
      "Ia-91T": 0.12
    }
  }
}
```

### Error Responses

**Status Code:** `400 Bad Request`

```json
{
  "detail": "No spectrum file or OSC reference provided"
}
```

**Status Code:** `500 Internal Server Error`

```json
{
  "detail": "Classification error: Model not found"
}
```

## Examples

### File Upload

#### cURL

```bash
curl -X POST "http://localhost:5000/process" \
  -F "file=@spectrum.fits" \
  -F 'params={"smoothing": 6, "knownZ": true, "zValue": 0.5, "calculateRlap": true}'
```

#### Python

```python
import requests

files = {'file': open('spectrum.fits', 'rb')}
data = {
    'params': '{"smoothing": 6, "knownZ": true, "zValue": 0.5}'
}

response = requests.post('http://localhost:5000/process',
                        files=files, data=data)
result = response.json()
print(f"Top match: {result['classification']['top_match']}")
```

#### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('params', JSON.stringify({
  smoothing: 6,
  knownZ: true,
  zValue: 0.5
}));

fetch('http://localhost:5000/process', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### OSC Reference

#### cURL

```bash
curl -X POST "http://localhost:5000/process" \
  -F 'params={"oscRef": "osc-sn2011fe-0", "smoothing": 4}'
```

#### Python

```python
import requests

data = {
    'params': '{"oscRef": "osc-sn2011fe-0", "smoothing": 4}'
}

response = requests.post('http://localhost:5000/process', data=data)
result = response.json()
```

## Processing Details

### Spectrum Processing Pipeline

1. **File Reading**: Supports FITS, DAT, TXT, and LNW formats
2. **Wavelength Range**: Optional filtering by min/max wavelength
3. **Smoothing**: Median filtering with configurable kernel size
4. **Redshift Correction**: Applies known redshift if provided
5. **Normalization**: Continuum removal and flux normalization
6. **Classification**: ML model prediction with confidence scores
7. **RLAP Calculation**: Optional relative likelihood analysis

### Classification Results

The classification includes:
- **Top Match**: Highest confidence supernova type
- **Confidence**: Probability score (0-1)
- **All Matches**: Complete list of predictions with scores
- **Age Bins**: Temporal classification within each type
- **RLAP Values**: Relative likelihood ratios (if calculated)

## Notes

- **File Size Limit**: 50MB per file
- **Processing Time**: 1-5 seconds depending on spectrum complexity
- **Supported Formats**: FITS, DAT, TXT, LNW
- **Wavelength Range**: 3500-10000 Angstroms (configurable)
- **Redshift Range**: 0.0-2.0 (for known redshifts)
