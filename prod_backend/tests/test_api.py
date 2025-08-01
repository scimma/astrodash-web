import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "AstroDash API" in response.json()["message"]

def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "warning", "unhealthy"]
    assert "timestamp" in data
    assert "version" in data
    assert "metrics" in data

def test_line_list_elements():
    """Test the line list elements endpoint."""
    response = client.get("/api/v1/line-list/elements")
    assert response.status_code == 200
    data = response.json()
    assert "elements" in data
    assert isinstance(data["elements"], list)
    assert len(data["elements"]) > 0

def test_analysis_options():
    """Test the analysis options endpoint."""
    response = client.get("/api/v1/analysis-options")
    assert response.status_code == 200
    data = response.json()
    # The endpoint returns a dictionary with age_bins_by_type and sn_types
    assert isinstance(data, dict)
    assert "age_bins_by_type" in data
    assert "sn_types" in data
    assert isinstance(data["sn_types"], list)
    assert len(data["sn_types"]) > 0
    assert isinstance(data["age_bins_by_type"], dict)

def test_osc_references():
    """Test the OSC references endpoint."""
    response = client.get("/api/v1/osc-references")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "references" in data
    assert data["status"] == "success"

def test_api_documentation():
    """Test that API documentation is available."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text

def test_openapi_spec():
    """Test that OpenAPI specification is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
