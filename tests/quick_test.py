# tests/quick_test.py
"""
Quick test script for the Trip Duration Predictor API.
This script sends a test request to the API using the reference values
and prints the results.

Usage:
    python quick_test.py

Requirements:
    requests
"""

import requests
import json
from datetime import datetime

# API endpoint URL - change this as needed
API_URL = "http://localhost:8000"

def main():
    """Main function to test the API"""
    print("Testing Trip Duration Predictor API...")
    
    # Sample test data based on the reference values
    test_data = {
        "vendor_id": 1,
        "passenger_count": 1,
        "pickup_datetime": "2023-05-16T12:00:00",
        "pickup_longitude": -73.9812,
        "pickup_latitude": 40.7648,
        "dropoff_longitude": -73.9708,
        "dropoff_latitude": 40.7617
    }
    
    # Expected results from the reference
    expected_duration = 12348  # seconds
    
    print("\n1. Testing API health...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ Health check successful!")
            print(f"   Status: {response.json()}")
        else:
            print(f"❌ Health check failed! Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed with exception: {str(e)}")
    
    print("\n2. Testing prediction endpoint...")
    try:
        print(f"   Request payload: {json.dumps(test_data, indent=2)}")
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction successful!")
            print(f"   Prediction: {data.get('prediction')} seconds")
            print(f"   Formatted: {data.get('formatted_time')}")
            
            # Compare with expected
            diff = abs(data.get('prediction', 0) - expected_duration)
            if diff <= 60:  # Within 60 seconds
                print(f"   ✅ Prediction matches reference (within 60s): {diff}s difference")
            else:
                print(f"   ⚠️ Prediction differs from reference: {diff}s difference")
        else:
            print(f"❌ Prediction failed! Status code: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Prediction failed with exception: {str(e)}")
    
    print("\n3. Testing frontend serving...")
    try:
        response = requests.get(API_URL)
        if response.status_code == 200 and "text/html" in response.headers.get("content-type", ""):
            print("✅ Frontend is being served correctly!")
        else:
            print(f"❌ Frontend serving failed! Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend check failed with exception: {str(e)}")

if __name__ == "__main__":
    main()