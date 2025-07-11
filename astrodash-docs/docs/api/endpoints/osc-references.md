---
sidebar_position: 8
---

# OSC References

Get available Open Supernova Catalog (OSC) references.

## Endpoint

```
GET /api/osc-references
```

## Description

Returns a list of available OSC references for use as spectrum sources.

## Request

No parameters required.

## Response

### Success Response
**Status Code:** `200 OK`

```json
{
  "status": "success",
  "references": [
    "osc-sn2002er-0",
    "osc-sn2011fe-0",
    ...
  ]
}
```

### Error Response
**Status Code:** `500 Internal Server Error`

```json
{
  "detail": "Error fetching OSC references: ..."
}
```

## Example

### cURL
```bash
curl -X GET "http://localhost:5000/api/osc-references"
```

### Python
```python
import requests
response = requests.get('http://localhost:5000/api/osc-references')
print(response.json())
```

## Notes
- Use these references as the `oscRef` parameter in the `/process` endpoint.
