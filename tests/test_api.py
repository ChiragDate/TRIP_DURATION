import pytest
from fastapi.testclient import TestClient
import json
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np

# Import the application
from src.service import app, PredictionInput

# Create a test client
client = TestClient(app)

# Mock the model to avoid loading the actual model during tests
mock_model = MagicMock()
mock_model.predict.return_value = np.array([1234.56])

@pytest.fixture
def sample_input_data():
    """Fixture to provide sample prediction input data."""
    return {
        "vendor_id": 1.0,
        "passenger_count": 1.0,
        "pickup_longitude": -73.98215,
        "pickup_latitude": 40.762,
        "dropoff_longitude": -73.9583,
        "dropoff_latitude": 40.7152,
        "store_and_fwd_flag": 0.0,
        "distance_haversine": 2.0,
        "distance_dummy_manhattan": 2.3,
        "direction": 155.0,
        "pickup_weekday": 1.0,
        "pickup_hour": 14.0,
        "pickup_minute": 30.0,
        "pickup_dt": 1580000000.0,
        "pickup_week_hour": 38.0
    }

@pytest.fixture
def sample_input_object(sample_input_data):
    """Fixture to convert sample data to PredictionInput object."""
    return PredictionInput(**sample_input_data)

@pytest.fixture(autouse=True)
def setup_mock_model():
    """Patch the model for all tests."""
    with patch('src.service.model', mock_model):
        yield

def test_api_home_endpoint():
    """Test the API home endpoint."""
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json() == {"message": "Trip Duration Prediction API is running"}

def test_root_endpoint():
    """Test the root endpoint which serves the UI."""
    with patch('os.path.exists', return_value=True):
        with patch('fastapi.responses.FileResponse', return_value=MagicMock(status_code=200)):
            response = client.get("/")
            assert response.status_code == 200

def test_health_check_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_predict_endpoint(sample_input_data):
    """Test the prediction endpoint with valid data."""
    response = client.post(
        "/predict",
        json=sample_input_data
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)
    assert "prediction_time_ms" in response.json()

def test_predict_processing(sample_input_object):
    """Test the prediction function correctly processes input data."""
    mock_model.predict.return_value = np.array([987.65])

    with patch('src.service.model', mock_model):
        mock_model.predict.reset_mock()  # Reset the call count

        response = client.post(
            "/predict",
            json=sample_input_object.model_dump()  
        )

        # Verify model was called with correct parameters
        mock_model.predict.assert_called_once()
        args, _ = mock_model.predict.call_args
        
        # Check the input feature list length matches what the model expects
        assert len(args[0][0]) == 15
        assert args[0][0][0] == sample_input_object.vendor_id
        assert args[0][0][1] == sample_input_object.passenger_count
        assert response.json()["prediction"] == 987.65

def test_predict_with_invalid_data():
    """Test the prediction endpoint with invalid data."""
    invalid_data = {
        "vendor_id": "invalid",  # Should be a float
        "passenger_count": 1.0,
        "pickup_longitude": -73.98215,
        "pickup_latitude": 40.762,
        "dropoff_longitude": -73.9583,
        "dropoff_latitude": 40.7152
        # Missing other required fields
    }
    
    response = client.post(
        "/predict",
        json=invalid_data
    )
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_with_empty_request():
    """Test the prediction endpoint with empty data."""
    response = client.post(
        "/predict",
        json={}
    )
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_with_extreme_values(sample_input_data):
    """Test the prediction endpoint with extreme values."""
    extreme_data = sample_input_data.copy()
    extreme_data["distance_haversine"] = 1000.0  # Very extreme value
    
    response = client.post(
        "/predict",
        json=extreme_data
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_metrics_endpoint():
    """Test the metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_name" in response.json()
    assert "model_version" in response.json()
    assert "uptime" in response.json()
    assert "status" in response.json()

def test_model_loading_error():
    """Test handling of model loading errors."""
    with pytest.raises(Exception):
        with patch('src.service.load', side_effect=Exception("Model not found")):
            from src.service import startup_event
            startup_event()

if __name__ == "__main__":
    pytest.main(["-xvs"])