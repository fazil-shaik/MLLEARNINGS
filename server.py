# server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load your trained model
model = joblib.load('your_model.pkl')

class PredictionRequest(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make prediction and return result"""
    try:
        # Convert to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        
        return {
            "prediction": prediction.tolist(),
            "probability": probability.tolist() if probability is not None else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-and-plot")
async def predict_and_plot(request: PredictionRequest):
    """Make prediction and return matplotlib plot as image"""
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Feature importance or distribution
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            ax1.bar(range(len(importances)), importances)
            ax1.set_title('Feature Importances')
            ax1.set_xlabel('Feature Index')
            ax1.set_ylabel('Importance')
        
        # Plot 2: Prediction result
        ax2.bar(['Prediction'], [prediction], color=['green' if prediction > 0.5 else 'red'])
        ax2.set_title(f'Prediction Result: {prediction:.2f}')
        ax2.set_ylabel('Value')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-with-plot-base64")
async def predict_with_plot_base64(request: PredictionRequest):
    """Return prediction and plot as base64 string"""
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Example: Show prediction confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            ax.bar(range(len(proba)), proba)
            ax.set_title('Prediction Probabilities')
            ax.set_xlabel('Class')
            ax.set_ylabel('Probability')
        else:
            ax.text(0.5, 0.5, f'Prediction: {prediction:.2f}', 
                   ha='center', va='center', fontsize=20)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return {
            "prediction": float(prediction),
            "plot": plot_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)