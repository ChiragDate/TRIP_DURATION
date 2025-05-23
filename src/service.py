import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import joblib
import pathlib
from datetime import datetime

load_dotenv()

model_path = 'models/model.joblib' 
model = joblib.load(model_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371 
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_minute'] = df['pickup_datetime'].dt.minute
    
    base_date = pd.to_datetime('2016-01-01 00:00:00')
    df['pickup_dt'] = (df['pickup_datetime'] - base_date).dt.total_seconds()
    df['pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']
    
    df['distance_haversine'] = haversine_array(
        df['pickup_latitude'].values, 
        df['pickup_longitude'].values, 
        df['dropoff_latitude'].values, 
        df['dropoff_longitude'].values
    )
    
    df['distance_dummy_manhattan'] = dummy_manhattan_distance(
        df['pickup_latitude'].values, 
        df['pickup_longitude'].values, 
        df['dropoff_latitude'].values, 
        df['dropoff_longitude'].values
    )
    
    df['direction'] = bearing_array(
        df['pickup_latitude'].values, 
        df['pickup_longitude'].values, 
        df['dropoff_latitude'].values, 
        df['dropoff_longitude'].values
    )
    
    required_features = [
        'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
        'distance_haversine', 'distance_dummy_manhattan', 'direction',
        'pickup_weekday', 'pickup_hour', 'pickup_minute', 'pickup_dt', 'pickup_week_hour'
    ]
    
    return df[required_features]

@app.get("/", response_class=FileResponse)
async def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        
        input_dict = {
            "vendor_id": int(data.get("vendor_id", 1)),
            "passenger_count": int(data.get("passenger_count", 1)),
            "pickup_datetime": data.get("pickup_datetime"),
            "dropoff_datetime": data.get("dropoff_datetime"), 
            "pickup_longitude": float(data.get("pickup_longitude")),
            "pickup_latitude": float(data.get("pickup_latitude")),
            "dropoff_longitude": float(data.get("dropoff_longitude")),
            "dropoff_latitude": float(data.get("dropoff_latitude")),
            "store_and_fwd_flag": 0,  
            "trip_duration": 0  
        }
        
        if input_dict['pickup_datetime'] and 'T' in input_dict['pickup_datetime']:
            input_dict['pickup_datetime'] = input_dict['pickup_datetime'].replace('T', ' ')
        
        df_features = preprocess_input(input_dict)
        
        prediction = model.predict(df_features)[0]
        
        prediction_seconds = int(round(prediction))
        minutes = prediction_seconds // 60
        seconds = prediction_seconds % 60
        
        return JSONResponse(content={
            "prediction": prediction_seconds,
            "formatted_time": f"{minutes} minutes and {seconds} seconds"
        })
        
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500, 
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )