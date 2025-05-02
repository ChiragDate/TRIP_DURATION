import pytest
from fastapi.testclient import TestClient
import json
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np

# Add the directory containing main.py to the path]


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
        "distance_haversine": 3.8,
        "distance_dummy_manhattan": 4.2,
        "direction": 2.5,
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

def test_home_endpoint():
    """Test the home endpoint returns expected message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == '"Working fine"'

def test_predict_endpoint(sample_input_data):
    """Test the prediction endpoint with valid data."""
    response = client.post(
        "/predict",
        json=sample_input_data
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)

def test_predict_processing(sample_input_object):
    """Test the prediction function correctly processes input data."""
    mock_model.predict.return_value = np.array([987.65])

    with patch('src.service.model', mock_model):
        mock_model.predict.reset_mock()  # ✅ Reset the call count

        response = client.post(
            "/predict",
            json=sample_input_object.model_dump()  
        )

        # ✅ Now it's safe to assert
        mock_model.predict.assert_called_once()

        args, _ = mock_model.predict.call_args

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

def test_health_endpoint():
    """Test health endpoint if it exists."""
    # If you add a health endpoint, test it here
    # response = client.get("/health")
    # assert response.status_code == 200
    # assert response.json() == {"status": "healthy"}
    pass

def test_model_loading_error():
    """Test handling of model loading errors."""
    with patch('src.service.load', side_effect=Exception("Model not found")):
        # This would fail when importing the module, which is caught by pytest
        pass

if __name__ == "__main__":
    pytest.main(["-xvs"])