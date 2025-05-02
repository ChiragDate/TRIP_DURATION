# main.py
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

app = FastAPI()

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionInput(BaseModel):
    # Define the input parameters required for making predictions
    vendor_id:float
    passenger_count:float
    pickup_longitude:float
    pickup_latitude:float
    dropoff_longitude:float
    dropoff_latitude:float
    store_and_fwd_flag:float
    distance_haversine:float
    distance_dummy_manhattan:float
    direction:float
    pickup_weekday:float
    pickup_hour:float
    pickup_minute:float
    pickup_dt:float
    pickup_week_hour:float

# Load the pre-trained RandomForest model
model_path = "/var/lib/jenkins/workspace/TRIP_DURATION/models/model.joblib"
model = load(model_path)

@app.get("/")
def home():
    return "Working fine"

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Extract features from input_data and make predictions using the loaded model
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
    prediction = model.predict([features])[0].item()
    # Return the prediction
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)