---
sidebar_position: 1
---

# Welcome to the AstroDash API


AstroDash is a API for supernova spectrum classification and redshift estimation using Machine Learning.

- **Fast, accurate classification** (DASH CNN, Transformer, user models)
- **Redshift estimation** using validated templates
- **Single and batch processing** with file and OSC reference inputs
- **Strong contracts** with versioned REST endpoints and thorough docs

Explore the docs to get started, or jump to the [Interactive API Explorer](swagger). The v1 API lives under the `/api/v1` prefix.

---

## Purpose and Audience

AstroDash serves astronomers, astrophysicists, machine learning researchers, and tool builders who need robust, reproducible classification of supernova spectra and related analysis. The API is optimized for interactive web apps and automated pipelines alike.

## Base URL and Versioning

- Base URL (local): `http://localhost:8000`
- API prefix: `/api/v1`
- Versioning policy: breaking changes are introduced only in new major versions (e.g., `/api/v2`). Backward-compatible changes can ship in minor versions.

## Quick Start

1) Health check: `GET /health`
2) Classify one spectrum: `POST /api/v1/process` with `file` (or `params.oscRef`)
3) Explore and test endpoints: Swagger UI at `/docs`, OpenAPI at `/openapi.json`

## API Surface Summary

- **Spectrum and Classification**
  - `POST /api/v1/process`: classify a single spectrum (file or OSC reference)
- **Batch**
  - `POST /api/v1/batch-process`: classify a ZIP folder or multiple files
- **Templates**
  - `GET /api/v1/analysis-options`: list valid SN types and age bins for DASH
  - `GET /api/v1/template-spectrum`: return a DASH template spectrum (`x`, `y`)
- **Line Lists**
  - `GET /api/v1/line-list[...]`: Get a list of single element or compound wavelength markers
- **User Models**
  - `POST /api/v1/upload-model`: upload TorchScript model with metadata
  - `GET/PUT/DELETE /api/v1/models/...`: manage models
- **Redshift**
  - `POST /api/v1/estimate-redshift`: estimate redshift (using DASH templates). More ways of estimating coming soon.

## Conventions and Limits

- Content types: multipart/form-data for uploads; JSON responses
- File formats: FITS, DAT, TXT, LNW, CSV
- Units: wavelength in Å; flux normalized 0 to 1
- Timeouts and limits: typical requests complete within seconds; see batch docs for guidance

## Errors

All errors return JSON with a `detail` message and appropriate HTTP status codes. See the [Error Model](/docs/api/errors) for canonical shapes and examples.

## Security and Access

- CORS is enabled for API usage; no authentication is required in local development
- Security headers and rate limiting are applied by default

## Environments

- Local development: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

## Next Steps

- [Getting Started Guide](/docs/guides/getting-started)
- [Architecture Overview](/docs/api/architecture-overview)
- [Error Model](/docs/api/errors)
- [API Endpoints Reference](/docs/api/endpoints/health)
- [Code Examples](/docs/guides/code-examples/python)
- [Interactive API Explorer](swagger)
