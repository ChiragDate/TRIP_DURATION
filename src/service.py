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

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        values = [data.get("vendor_id"), data.get("passenger_count"),
                  data.get("trip_distance"), data.get("payment_type"),
                  data.get("fare_amount")]

        # Reshape input as expected by the model
        prediction = model.predict([values])[0]

        return JSONResponse(content={"predicted_duration": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
