# src/service.py

import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import joblib
from feature_definitions import feature_build
from datetime import datetime


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

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

print("="*30)
print("="*30)
print("="*30)
print(f"STATIC_DIR: {STATIC_DIR}")
print("="*30)
print("="*30)
print("="*30)

@app.get("/", response_class=FileResponse)
async def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

    @app.get("/health")
    def health_check():
        return {"status": "ok"}
    
    
    do_not_train = [
    'id', 'pickup_datetime', 'dropoff_datetime', 'check_trip_duration',
    'pickup_date', 'avg_speed_h', 'avg_speed_m',
    'pickup_lat_bin', 'pickup_long_bin', 'center_lat_bin', 'center_long_bin',
    'pickup_dt_bin', 'pickup_datetime_group'
]

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()

        input_dict = {
            "id": 0,
            "vendor_id": data.get("vendor_id", 1),
            "pickup_datetime": data.get("pickup_datetime"),
            "dropoff_datetime": data.get("dropoff_datetime"),
            "passenger_count": data.get("passenger_count", 1),
            "pickup_longitude": data.get("pickup_longitude", 0.0),
            "pickup_latitude": data.get("pickup_latitude", 0.0),
            "dropoff_longitude": data.get("dropoff_longitude", 0.0),
            "dropoff_latitude": data.get("dropoff_latitude", 0.0),
            "store_and_fwd_flag": 0,
            "trip_duration": 0
        }

        if input_dict['pickup_datetime'] and 'T' in input_dict['pickup_datetime']:
            input_dict['pickup_datetime'] = input_dict['pickup_datetime'].replace('T', ' ')

        if input_dict['dropoff_datetime'] and 'T' in input_dict['dropoff_datetime']:
            input_dict['dropoff_datetime'] = input_dict['dropoff_datetime'].replace('T', ' ')

        df = pd.DataFrame([input_dict])
        df_feat = feature_build(df)

        features = [f for f in df_feat.columns if f not in do_not_train]

        prediction = model.predict(df_feat[features])[0]

        return JSONResponse(content={"predicted_duration": float(prediction)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
