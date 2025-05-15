# src/service.py

import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import joblib

# Load environment variables
load_dotenv()

# Load the model
model_path = 'models/model.joblib'  # Adjust the path as necessary
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

STATIC_DIR = os.path.join(os.getcwd(), "static")
print(f"Static files directory: {STATIC_DIR}")

@app.get("/", response_class=FileResponse)
async def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()

        feature_keys = [
            "vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude", "fare_amount", "extra",
            "mta_tax", "pickup_weekday", "pickup_hour", "pickup_minute",
            "pickup_dt", "pickup_week_hour", "store_and_fwd_flag",
            "distance_haversine", "distance_dummy_manhattan", "direction"
        ]

        # Ensure all keys are present
        values = [data.get(key, 0.0) for key in feature_keys]

        prediction = model.predict([values])[0]

        return JSONResponse(content={"predicted_duration": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

