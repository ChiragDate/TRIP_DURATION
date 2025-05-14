# test_api.py

import pytest
from httpx import AsyncClient
from src.service import app


@pytest.mark.asyncio
async def test_root_serves_index_or_404():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        assert response.headers["content-type"].startswith("text/html")
    else:
        assert "index.html not found" in response.text

@pytest.mark.asyncio
async def test_predict_valid_input(monkeypatch):
    # Mock model's predict function
    def mock_predict(input_data):
        return [12.34]  # Dummy prediction

    monkeypatch.setattr("src.service.model.predict", mock_predict)

    payload = {
        "vendor_id": 1,
        "passenger_count": 2,
        "trip_distance": 5.0,
        "payment_type": 1,
        "fare_amount": 15.50
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_duration" in response.json()

@pytest.mark.asyncio
async def test_predict_invalid_input():
    payload = {
        "vendor_id": 1,
        # Missing other required fields
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict", json=payload)
    assert response.status_code == 500
    assert "error" in response.json()

# test_api.py

@pytest.mark.asyncio
async def test_predict_full_feature_set(monkeypatch):
    def mock_predict(input_data):
        return [9.87]

    monkeypatch.setattr("src.service.model.predict", mock_predict)

    payload = {
        "vendor_id": 1.0,
        "passenger_count": 1.0,
        "pickup_longitude": -73.98215,
        "pickup_latitude": 40.762,
        "dropoff_longitude": -73.9583,
        "dropoff_latitude": 40.7152,
        "fare_amount": 3.8,
        "extra": 4.2,
        "mta_tax": 2.5,
        "pickup_weekday": 1.0,
        "pickup_hour": 14.0,
        "pickup_minute": 30.0,
        "pickup_dt": 1580000000.0,
        "pickup_week_hour": 38.0,
        "store_and_fwd_flag": 0.0,
        "distance_haversine": 2.0,
        "distance_dummy_manhattan": 2.3,
        "direction": 155.0
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert "predicted_duration" in response.json()
    assert isinstance(response.json()["predicted_duration"], float)
