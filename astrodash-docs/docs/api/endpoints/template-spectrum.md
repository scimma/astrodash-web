---
sidebar_position: 5
---

# Template Spectrum

Get a template spectrum for a given SN type and age bin.

## Endpoint

```
GET /api/template-spectrum
```

## Description

Returns a template spectrum for a specified supernova type and age bin. Useful for comparison and visualization.

## Request

### Query Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sn_type` | String | No | "Ia" | Supernova type (e.g., "Ia-norm") |
| `age_bin` | String | No | "2 to 6" | Age bin (e.g., "4 to 8") |

## Response

### Success Response
**Status Code:** `200 OK`

```json
{
  "wave": [3500.0, 3501.0, ...],
  "flux": [0.1, 0.2, ...],
  "sn_type": "Ia-norm",
  "age_bin": "4 to 8"
}
```

### Error Response
**Status Code:** `404 Not Found`

```json
{
  "detail": "Template data file not found."
}
```

## Example

### cURL
```bash
curl -X GET "http://localhost:5000/api/template-spectrum?sn_type=Ia-norm&age_bin=4%20to%208"
```

### Python
```python
import requests
params = {'sn_type': 'Ia-norm', 'age_bin': '4 to 8'}
response = requests.get('http://localhost:5000/api/template-spectrum', params=params)
print(response.json())
```

## Notes
- Use `/api/analysis-options` to get valid SN types and age bins.
- Returned arrays are suitable for plotting or further analysis.
