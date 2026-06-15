from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

MODEL_DIR = Path(__file__).resolve().parent / "models"
FEATURE_COLUMNS = [
    "roast_time_min",
    "temp_ramp_c_min",
    "moisture_pct",
    "density_g_ml",
    "airflow"
]

# Model version tracking
MODEL_VERSION = "1.0.0"
MODEL_INFO = {
    "linear": {"version": "1.0.0", "trained_date": "2024-01-15"},
    "lasso": {"version": "1.0.0", "trained_date": "2024-01-15"},
    "ridge": {"version": "1.0.0", "trained_date": "2024-01-15"},
    "polynomial": {"version": "1.0.0", "trained_date": "2024-01-15"}
}

# Load models at startup
models = {}
feature_columns = FEATURE_COLUMNS.copy()

async def load_models():
    global models, feature_columns
    try:
        with open(MODEL_DIR / 'linear.pkl', 'rb') as f:
            models['linear'] = pickle.load(f)
        with open(MODEL_DIR / 'lasso.pkl', 'rb') as f:
            models['lasso'] = pickle.load(f)
        with open(MODEL_DIR / 'ridge.pkl', 'rb') as f:
            models['ridge'] = pickle.load(f)
        with open(MODEL_DIR / 'polynomial.pkl', 'rb') as f:
            models['polynomial'] = pickle.load(f)
        try:
            with open(MODEL_DIR / 'feature_columns.pkl', 'rb') as f:
                feature_columns = pickle.load(f)
        except FileNotFoundError:
            print('Feature columns file not found; using default feature order.')
        print("Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
    except Exception as e:
        print(f"Error loading models: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models()
    yield

app = FastAPI(
    title="Coffee Roast Predictor API",
    description="Predict coffee quality metrics using multiple regression models",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class RoastParameters(BaseModel):
    roast_time_min: float
    temp_ramp_c_min: float
    moisture_pct: float
    density_g_ml: float
    airflow: int
    
    class Config:
        schema_extra = {
            "example": {
                "roast_time_min": 11.5,
                "temp_ramp_c_min": 5.2,
                "moisture_pct": 10.5,
                "density_g_ml": 0.72,
                "airflow": 6
            }
        }

class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    acidity_prediction: float
    timestamp: str
    feature_importance: Dict[str, float] = None

class AllPredictionsResponse(BaseModel):
    predictions: Dict[str, float]
    model_versions: Dict[str, str]
    input_parameters: Dict
    timestamp: str

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Coffee Roast Predictor API",
        "version": MODEL_VERSION,
        "available_models": list(models.keys()),
        "endpoints": ["/predict/{model_name}", "/predict_all", "/models", "/health"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def list_models():
    return {
        "models": MODEL_INFO,
        "feature_columns": feature_columns,
        "total_models": len(models)
    }

@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict_single(model_name: str, params: RoastParameters):
    
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Prepare input data
    input_data = pd.DataFrame([[
        params.roast_time_min,
        params.temp_ramp_c_min,
        params.moisture_pct,
        params.density_g_ml,
        params.airflow
    ]], columns=feature_columns)
    
    try:
        prediction = models[model_name].predict(input_data)[0]
        
        feature_importance = None
        if model_name in ['linear', 'lasso', 'ridge']:
            feature_importance = dict(zip(feature_columns, models[model_name].coef_))
        
        return PredictionResponse(
            model_name=model_name,
            model_version=MODEL_INFO.get(model_name, {}).get("version", "unknown"),
            acidity_prediction=float(np.clip(prediction, 1, 10)),  # Keep within 1-10 scale
            timestamp=datetime.now().isoformat(),
            feature_importance=feature_importance
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_all", response_model=AllPredictionsResponse)
async def predict_all(params: RoastParameters):
    """Get predictions from ALL models for comparison"""
    
    input_data = pd.DataFrame([[
        params.roast_time_min,
        params.temp_ramp_c_min,
        params.moisture_pct,
        params.density_g_ml,
        params.airflow
    ]], columns=feature_columns)
    
    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(input_data)[0]
            predictions[name] = float(np.clip(pred, 1, 10))
        except Exception as e:
            predictions[name] = None
    
    return AllPredictionsResponse(
        predictions=predictions,
        model_versions={name: MODEL_INFO.get(name, {}).get("version", "unknown") 
                       for name in models.keys()},
        input_parameters=params.dict(),
        timestamp=datetime.now().isoformat()
    )

@app.post("/retrain")
async def retrain_models():
    return {
        "message": "Retraining endpoint - implement your training logic here",
        "note": "This would trigger train_models.py and reload models"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)