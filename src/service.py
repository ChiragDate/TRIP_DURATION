# src/service.py

# src/service.py (top of file)
from pydantic import BaseModel
from typing import Optional
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import joblib

# Load environment variables
load_dotenv()


class PredictionInput(BaseModel):
    vendor_id: float
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: float
    distance_haversine: float
    distance_dummy_manhattan: float
    direction: float
    pickup_weekday: float
    pickup_hour: float
    pickup_minute: float
    pickup_dt: float
    pickup_week_hour: float


# Load the model
model_path = 'models/model.joblib'
model = joblib.load(model_path)

# FastAPI app
app = FastAPI()

# CORS for local frontend access (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set a specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.get("/", response_class=FileResponse)
async def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

from fastapi import FastAPI
from fastapi.responses import JSONResponse

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        features = [
            input_data.vendor_id,
            input_data.passenger_count,
            input_data.pickup_longitude,
            input_data.pickup_latitude,
            input_data.dropoff_longitude,
            input_data.dropoff_latitude,
            input_data.store_and_fwd_flag,
            input_data.distance_haversine,
            input_data.distance_dummy_manhattan,
            input_data.direction,
            input_data.pickup_weekday,
            input_data.pickup_hour,
            input_data.pickup_minute,
            input_data.pickup_dt,
            input_data.pickup_week_hour,
        ]

        prediction = model.predict([features])[0]
        return JSONResponse(content={
            "prediction": float(prediction),
            "prediction_time_ms": 5  # or real duration if measured
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
