# tests/test_api.py

import unittest
import requests
import json
import os
import sys
from datetime import datetime

# Add project root to path for imports if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# API endpoint URL - change this as needed
API_URL = "http://localhost:8000"

class TestTripDurationAPI(unittest.TestCase):
    """Test cases for the Trip Duration Predictor API"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Sample test data based on the reference values
        self.sample_data = {
            "vendor_id": 1,
            "passenger_count": 1,
            "pickup_datetime": "2023-05-16T12:00:00",
            "pickup_longitude": -73.9812,
            "pickup_latitude": 40.7648,
            "dropoff_longitude": -73.9708,
            "dropoff_latitude": 40.7617
        }
        
        # Expected results from the reference
        self.expected_duration = 478
        self.expected_minutes = 7
        self.expected_seconds = 58  # Adjust as per model's actual formatted_time output


    def test_api_health(self):
        """Test if the API health endpoint is responding"""
        response = requests.get(f"{API_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    def test_prediction_with_sample_data(self):
        """Test prediction with sample data matching the reference values"""
        response = requests.post(
            f"{API_URL}/predict",
            json=self.sample_data
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check if prediction field exists
        self.assertIn("prediction", data)
        
        # Check if the value is close to the expected value
        # Using a tolerance range since model outputs might vary slightly
        self.assertAlmostEqual(
            data["prediction"], 
            self.expected_duration,
            delta=60  # Allow up to 60 seconds difference
        )
        
        # Check formatted time
        self.assertIn("formatted_time", data)
        formatted_time = data["formatted_time"]
        
        # Verify the formatted time contains the expected minutes and seconds
        self.assertIn(str(self.expected_minutes), formatted_time)
        self.assertIn(str(self.expected_seconds), formatted_time)

    def test_prediction_with_missing_fields(self):
        """Test prediction with missing required fields"""
        # Remove a required field (pickup_longitude)
        incomplete_data = self.sample_data.copy()
        del incomplete_data["pickup_longitude"]
        
        response = requests.post(
            f"{API_URL}/predict",
            json=incomplete_data
        )
        
        # Should return an error
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("error", data)

    def test_prediction_with_invalid_data_types(self):
        """Test prediction with invalid data types"""
        # Use string instead of number for passenger_count
        invalid_data = self.sample_data.copy()
        invalid_data["passenger_count"] = "not_a_number"
        
        response = requests.post(
            f"{API_URL}/predict",
            json=invalid_data
        )
        
        # Should return an error
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("error", data)

    def test_prediction_with_extreme_values(self):
        """Test prediction with extreme but valid values"""
        # Use extreme values for coordinates (still within NYC area)
        extreme_data = self.sample_data.copy()
        extreme_data["pickup_longitude"] = -74.05
        extreme_data["pickup_latitude"] = 40.85
        extreme_data["dropoff_longitude"] = -73.75
        extreme_data["dropoff_latitude"] = 40.65
        
        response = requests.post(
            f"{API_URL}/predict",
            json=extreme_data
        )
        
        # Should still work, even with extreme values
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        
        # Prediction should be a positive number
        self.assertTrue(data["prediction"] > 0)

    def test_prediction_with_optional_fields(self):
        """Test prediction with optional fields included"""
        # Add dropoff_datetime
        data_with_optional = self.sample_data.copy()
        data_with_optional["dropoff_datetime"] = "2023-05-16T15:30:00"
        
        response = requests.post(
            f"{API_URL}/predict",
            json=data_with_optional
        )
        
        # Should work with optional fields
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)

    def test_prediction_batch(self):
        """Test multiple predictions to check consistency"""
        results = []
        
        # Make 3 identical requests and store results
        for _ in range(3):
            response = requests.post(
                f"{API_URL}/predict",
                json=self.sample_data
            )
            self.assertEqual(response.status_code, 200)
            results.append(response.json()["prediction"])
        
        # All predictions should be identical (deterministic model)
        self.assertEqual(len(set(results)), 1, "Predictions should be consistent for identical inputs")

    def test_frontend_serving(self):
        """Test if the frontend is being served correctly"""
        response = requests.get(API_URL)
        self.assertEqual(response.status_code, 200)
        # Check if content is HTML
        self.assertIn("text/html", response.headers.get("content-type", ""))
        # Basic check for HTML content
        self.assertIn("<html", response.text.lower())
        self.assertIn("trip duration predictor", response.text.lower())


if __name__ == "__main__":
    unittest.main()