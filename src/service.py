import logging
import os
import time
import shutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load

# Set the log file path manually
log_path = "/var/lib/jenkins/workspace/TRIP_DURATION/logs/trip-duration-api.log"
log_dir = os.path.dirname(log_path)

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode='a+')
    ]
)

logger = logging.getLogger(__name__)

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

# Define application directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create necessary directories
os.makedirs(STATIC_DIR, exist_ok=True)

# Setup FastAPI app
app = FastAPI(
    title="Trip Duration Prediction API",
    description="API for predicting trip durations using machine learning models",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

model = None

@app.on_event("startup")
async def startup_event():
    """Startup event to load model"""
    global model
    
    # Create index.html in static directory if it doesn't exist
    index_html_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_html_path):
        logger.warning(f"Frontend file not found at: {index_html_path}")
        logger.warning("Please add the HTML file to the static directory")
    
    default_path = os.path.join(os.getcwd(), "models/model.joblib")
    model_path = os.getenv("MODEL_PATH", default_path)
    logger.info(f"Loading model from {model_path}")
    try:
        model = load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

@app.get("/", response_class=FileResponse)
async def read_root():
    """Serves the Trip Duration Prediction UI"""
    try:
        index_html_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_html_path):
            return FileResponse(index_html_path)
        else:
            return HTMLResponse(content="<html><body><h1>UI not found</h1><p>Please add index.html to the static directory</p></body></html>")
    except Exception as e:
        logger.error(f"Failed to serve UI: {str(e)}")
        return HTMLResponse(content=f"<html><body><h1>Error loading UI</h1><p>{str(e)}</p></body></html>")

@app.get("/api")
def api_home():
    """API information endpoint"""
    return {"message": "Trip Duration Prediction API is running"}

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring and liveness probes"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "status": "healthy",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Make a trip duration prediction based on the input data
    """
    logger.info("Received prediction request")
    try:
        features = [input_data.vendor_id,
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
                    input_data.pickup_week_hour
                    ]
        
        logger.debug(f"Input features: {features}")
        
        start_time = time.time()
        prediction = model.predict([features])[0].item()
        prediction_time = time.time() - start_time
        
        logger.info(f"Prediction: {prediction}, Time taken: {prediction_time:.4f}s")
        
        return {"prediction": prediction, "prediction_time_ms": prediction_time * 1000}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
def metrics():
    """
    Endpoint for monitoring metrics
    """
    return {
        "model_name": "trip_duration_predictor",
        "model_version": "1.0.0",
        "uptime": time.time() - startup_time,
        "status": "operational"
    }

startup_time = time.time()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)