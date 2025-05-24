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
import logging
import time

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to stdout
        logging.FileHandler("/var/log/trip-duration-api.log")  # Log to file
    ]
)
logger = logging.getLogger(__name__)

# Load model with logging
model_path = 'models/model.joblib' 
logger.info(f"Loading model from {model_path}")
try:
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise e

app = FastAPI(
    title="Trip Duration Prediction API",
    description="API for predicting trip durations using machine learning models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

def haversine_array(lat1, lng1, lat2, lng2):
    """Calculate haversine distance between coordinates"""
    logger.debug("Calculating haversine distance")
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    """Calculate dummy Manhattan distance using haversine components"""
    logger.debug("Calculating dummy Manhattan distance")
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    """Calculate bearing between coordinates"""
    logger.debug("Calculating bearing")
    AVG_EARTH_RADIUS = 6371 
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def preprocess_input(input_data):
    logger.info("Starting input data preprocessing")
    try:
        df = pd.DataFrame([input_data])
        
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        logger.debug(f"Parsed pickup datetime: {df['pickup_datetime'].iloc[0]}")
        
        df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_minute'] = df['pickup_datetime'].dt.minute
        
        base_date = pd.to_datetime('2016-01-01 00:00:00')
        df['pickup_dt'] = (df['pickup_datetime'] - base_date).dt.total_seconds()
        df['pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']
        
        logger.debug(f"Extracted time features - weekday: {df['pickup_weekday'].iloc[0]}, hour: {df['pickup_hour'].iloc[0]}")
        
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
        
        logger.debug(f"Calculated distances - haversine: {df['distance_haversine'].iloc[0]:.2f}km, "
                    f"manhattan: {df['distance_dummy_manhattan'].iloc[0]:.2f}km, "
                    f"direction: {df['direction'].iloc[0]:.2f}°")
        
        required_features = [
            'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
            'distance_haversine', 'distance_dummy_manhattan', 'direction',
            'pickup_weekday', 'pickup_hour', 'pickup_minute', 'pickup_dt', 'pickup_week_hour'
        ]
        
        result_df = df[required_features]
        logger.info("Input data preprocessing completed successfully")
        logger.debug(f"Final feature shape: {result_df.shape}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise e

@app.get("/", response_class=FileResponse)
async def serve_ui():
    """Serve the main UI page"""
    logger.info("Serving UI page")
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        logger.debug(f"Serving index.html from {index_path}")
        return FileResponse(index_path)
    logger.warning(f"index.html not found at {index_path}")
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    logger.debug("Health check requested")
    try:
        if model is None:
            logger.error("Health check failed - model not loaded")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "Model not loaded"}
            )
        
        logger.debug("Health check passed")
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "model_loaded": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": str(e)}
        )

@app.post("/predict")
async def predict(request: Request):
    """Make trip duration prediction"""
    logger.info("Received prediction request")
    start_time = time.time()
    
    try:
        data = await request.json()
        logger.debug(f"Request data keys: {list(data.keys())}")
        
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
        
        logger.debug(f"Input coordinates - pickup: ({input_dict['pickup_latitude']}, {input_dict['pickup_longitude']}), "
                    f"dropoff: ({input_dict['dropoff_latitude']}, {input_dict['dropoff_longitude']})")
        
        if input_dict['pickup_datetime'] and 'T' in input_dict['pickup_datetime']:
            input_dict['pickup_datetime'] = input_dict['pickup_datetime'].replace('T', ' ')
            logger.debug(f"Formatted pickup datetime: {input_dict['pickup_datetime']}")
        
        preprocessing_start = time.time()
        df_features = preprocess_input(input_dict)
        preprocessing_time = time.time() - preprocessing_start
        logger.info(f"Preprocessing completed in {preprocessing_time:.4f}s")
    
        prediction_start = time.time()
        prediction = model.predict(df_features)[0]
        prediction_time = time.time() - prediction_start
        
        prediction_seconds = int(round(prediction))
        minutes = prediction_seconds // 60
        seconds = prediction_seconds % 60
        
        total_time = time.time() - start_time
        
        logger.info(f"Prediction completed - duration: {prediction_seconds}s ({minutes}m {seconds}s), "
                   f"prediction time: {prediction_time:.4f}s, total time: {total_time:.4f}s")
        
        return JSONResponse(content={
            "prediction": prediction_seconds,
            "formatted_time": f"{minutes} minutes and {seconds} seconds",
            "processing_time_ms": total_time * 1000,
            "prediction_time_ms": prediction_time * 1000
        })
        
    except ValueError as e:
        logger.error(f"Validation error in prediction: {str(e)}")
        return JSONResponse(
            status_code=400, 
            content={
                "error": "Invalid input data",
                "detail": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500, 
            content={
                "error": "Internal server error",
                "detail": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.get("/metrics")
def metrics():
    """Endpoint for monitoring metrics"""
    logger.debug("Metrics endpoint accessed")
    return {
        "model_name": "trip_duration_predictor",
        "model_version": "1.0.0",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up")
    logger.info(f"Static directory: {STATIC_DIR}")
    logger.info("Application startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting application with uvicorn")
    uvicorn.run(app, host="0.0.0.0", port=8000)