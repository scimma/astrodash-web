---
sidebar_position: 6
---

# Estimate Redshift

Estimate the redshift for a given spectrum and SN type/age.

## Endpoint

```
POST /api/estimate-redshift
```

## Description

Estimates the redshift of an input spectrum using template matching for a specified SN type and age bin.

## Request

### Content-Type
```
application/json
```

### JSON Body
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `x` | Array[float] | Yes | Wavelengths |
| `y` | Array[float] | Yes | Fluxes |
| `type` | String | Yes | SN type (e.g., "Ia-norm") |
| `age` | String | Yes | Age bin (e.g., "4 to 8") |

## Response

### Success Response
**Status Code:** `200 OK`

```json
{
  "estimated_redshift": 0.123,
  "estimated_redshift_error": 0.005,
  "message": "No valid templates found for this type/age."
}
```

## Example

### Python
```python
import requests
import numpy as np

wavelengths = np.linspace(3500, 10000, 1000)
fluxes = np.random.normal(1.0, 0.1, 1000)
data = {
    'x': wavelengths.tolist(),
    'y': fluxes.tolist(),
    'type': 'Ia-norm',
    'age': '4 to 8'
}
response = requests.post('http://localhost:5000/api/estimate-redshift', json=data)
print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:5000/api/estimate-redshift" \
  -H "Content-Type: application/json" \
  -d '{"x": [3500, 3501, ...], "y": [0.1, 0.2, ...], "type": "Ia-norm", "age": "4 to 8"}'
```

## Notes
- Use `/api/analysis-options` to get valid SN types and age bins.
- The `message` field is present if no valid templates are found.
