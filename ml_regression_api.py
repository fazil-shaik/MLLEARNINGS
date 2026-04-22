from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import json
from sklearn.datasets import make_regression

app = Flask(__name__)

# Models dictionary
models = {}
scalers = {}


def generate_data():
    try:
        print("📊 Generating synthetic data...")
        
        # Generate 1000 samples with 3 features
        X, y = make_regression(
            n_samples=1000,
            n_features=3,
            n_informative=3,
            noise=500,
            random_state=42
        )
        
        # Normalize X to realistic ranges
        # Feature 1: weight (0.2 - 5.0)
        X[:, 0] = ((X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())) * 4.8 + 0.2
        
        # Feature 2: depth (43 - 79)
        X[:, 1] = ((X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())) * 36 + 43
        
        # Feature 3: table (43 - 95)
        X[:, 2] = ((X[:, 2] - X[:, 2].min()) / (X[:, 2].max() - X[:, 2].min())) * 52 + 43
        
        # Normalize y to price range (1000 - 20000)
        y = ((y - y.min()) / (y.max() - y.min())) * 19000 + 1000
        y = np.maximum(y, 500)  # Ensure positive prices
        
        print(f" Generated {len(X)} samples")
        print(f"   X shape: {X.shape}, y range: [{y.min():.0f}, {y.max():.0f}]")
        
        return X, y
    
    except Exception as e:
        print(f"Error generating data: {e}")
        return None, None



@app.route('/train', methods=['POST'])
def train_models():
    """
    Train all regression models on generated data
    """
    try:
        # Generate data
        X, y = generate_data()
        
        if X is None:
            return jsonify({"error": "Failed to generate data"}), 400
        
        print(f"\n🤖 Training 8 models...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scalers['standard'] = scaler
        print("✓ Scaler fitted")
        
        # 1. Linear Regression
        models['linear'] = LinearRegression()
        models['linear'].fit(X_scaled, y)
        print("✓ Linear Regression")
        
        # 2. Ridge Regression (L2 regularization)
        models['ridge'] = Ridge(alpha=1.0)
        models['ridge'].fit(X_scaled, y)
        print("✓ Ridge")
        
        # 3. Lasso Regression (L1 regularization)
        models['lasso'] = Lasso(alpha=0.1)
        models['lasso'].fit(X_scaled, y)
        print("✓ Lasso")
        
        # 4. Polynomial Regression (degree 2)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_scaled)
        models['poly'] = LinearRegression()
        models['poly'].fit(X_poly, y)
        scalers['poly'] = poly
        print("✓ Polynomial (degree 2)")
        
        # 5. SVR (Support Vector Regression)
        models['svr'] = SVR(kernel='rbf', C=100, epsilon=10)
        models['svr'].fit(X_scaled, y)
        print("✓ SVR (RBF kernel)")
        
        # 6. Random Forest
        models['rf'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        models['rf'].fit(X_scaled, y)
        print("✓ Random Forest")
        
        # 7. Gradient Boosting
        models['gb'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        models['gb'].fit(X_scaled, y)
        print("✓ Gradient Boosting")
        
        # 8. Neural Network (MLP)
        models['mlp'] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        models['mlp'].fit(X_scaled, y)
        print("✓ Neural Network (MLP)")
        
        print(f"\n ALL {len(models)} MODELS TRAINED SUCCESSFULLY!\n")
        
        return jsonify({
            "status": "success",
            "message": f"All {len(models)} models trained on 1000 samples",
            "models": list(models.keys()),
            "features": ["weight (carats)", "depth (mm)", "table (%)"],
            "target": "price ($)",
            "data_type": "synthetic"
        }), 200
    
    except Exception as e:
        import traceback
        print(f"   Training error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        X_input = np.array(data.get('X', []))
        
        if X_input.size == 0:
            return jsonify({"error": "Provide X features as [[weight, depth, table]]"}), 400
        
        if len(models) == 0:
            return jsonify({"error": "Models not trained. POST /train first"}), 400
        
        # Ensure input is 2D
        if X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)
        
        # Scale input
        X_scaled = scalers['standard'].transform(X_input)
        
        predictions = {}
        
        # Linear models
        predictions['linear'] = float(models['linear'].predict(X_scaled)[0])
        predictions['ridge'] = float(models['ridge'].predict(X_scaled)[0])
        predictions['lasso'] = float(models['lasso'].predict(X_scaled)[0])
        
        # Polynomial
        X_poly = scalers['poly'].transform(X_scaled)
        predictions['polynomial'] = float(models['poly'].predict(X_poly)[0])
        
        # Non-linear models
        predictions['svr'] = float(models['svr'].predict(X_scaled)[0])
        predictions['random_forest'] = float(models['rf'].predict(X_scaled)[0])
        predictions['gradient_boosting'] = float(models['gb'].predict(X_scaled)[0])
        predictions['neural_network'] = float(models['mlp'].predict(X_scaled)[0])
        
        # Average ensemble
        avg_pred = np.mean(list(predictions.values()))
        predictions['ensemble_average'] = float(avg_pred)
        
        return jsonify({
            "status": "success",
            "input_features": {
                "weight_carats": float(X_input[0][0]),
                "depth_mm": float(X_input[0][1]),
                "table_percent": float(X_input[0][2])
            },
            "predictions": predictions,
            "unit": "$"
        }), 200
    
    except Exception as e:
        import traceback
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500



@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        data = request.json
        X_input = np.array(data.get('X', []))
        
        if X_input.size == 0:
            return jsonify({"error": "Provide X features"}), 400
        
        if len(models) == 0:
            return jsonify({"error": "Models not trained"}), 400
        
        X_scaled = scalers['standard'].transform(X_input)
        
        results = []
        for i, x in enumerate(X_scaled):
            x_reshaped = x.reshape(1, -1)
            x_poly = scalers['poly'].transform(x_reshaped)
            
            preds = {
                'linear': float(models['linear'].predict(x_reshaped)[0]),
                'ridge': float(models['ridge'].predict(x_reshaped)[0]),
                'lasso': float(models['lasso'].predict(x_reshaped)[0]),
                'polynomial': float(models['poly'].predict(x_poly)[0]),
                'svr': float(models['svr'].predict(x_reshaped)[0]),
                'random_forest': float(models['rf'].predict(x_reshaped)[0]),
                'gradient_boosting': float(models['gb'].predict(x_reshaped)[0]),
                'neural_network': float(models['mlp'].predict(x_reshaped)[0]),
            }
            preds['ensemble_average'] = float(np.mean(list(preds.values())))
            
            results.append({
                'sample_id': i,
                'features': {
                    'weight_carats': float(X_input[i][0]),
                    'depth_mm': float(X_input[i][1]),
                    'table_percent': float(X_input[i][2])
                },
                'predictions': preds
            })
        
        return jsonify({
            "status": "success",
            "total_samples": len(results),
            "results": results
        }), 200
    
    except Exception as e:
        import traceback
        print(f"Batch prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "models_count": len(models)
    }), 200



@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "api": "ML Regression API",
        "models": list(models.keys()) if models else [],
        "endpoints": {
            "POST /train": "Train all models",
            "POST /predict": "Single prediction",
            "POST /predict-batch": "Batch predictions",
            "GET /health": "Health check",
            "GET /info": "API info"
        }
    }), 200


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ML REGRESSION API SERVER")
    print("="*60)
    print("Start training: POST /train")
    print("Make predictions: POST /predict")
    print("="*60 + "\n")
    app.run(debug=False, port=5000, threaded=True)