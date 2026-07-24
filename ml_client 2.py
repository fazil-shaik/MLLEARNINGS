import requests
import json
import numpy as np

# API Base URL
API_BASE = "http://127.0.0.1:5000"


def health_check():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE}/health")
        result = response.json()
        print(f"API running | Models loaded: {result.get('models_loaded', False)}")
        return result
    except Exception as e:
        print(f"API unreachable: {e}")
        return None


def train_all_models():
    """Train all regression models on generated data"""
    try:
        print("\nTraining all models...")
        response = requests.post(f"{API_BASE}/train")
        result = response.json()
        
        if response.status_code != 200:
            print(f"Training failed: {result.get('error', 'Unknown error')}")
            return None
        
        print(f"Status: {result.get('message', '')}")
        print(f"   Models: {', '.join(result.get('models', []))}")
        print(f"   Features: {result.get('features', [])}")
        print(f"   Target: {result.get('target', '')}\n")
        return result
    
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None



def predict_single(weight, depth, table):

    try:
        payload = {
            "X": [[weight, depth, table]]
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        
        if response.status_code != 200:
            print(f"Prediction failed: {result.get('error', 'Unknown error')}")
            return None
        
        print(f"\n SINGLE PREDICTION")
        print(f"Input: Weight={weight:.2f} carats, Depth={depth:.1f}mm, Table={table:.1f}%")
        print("\nModel Predictions:")
        
        predictions = result.get('predictions', {})
        if not predictions:
            print(f"No predictions in response")
            return None
        
        # Sort by ensemble first, then alphabetically
        sorted_preds = sorted(predictions.items(), key=lambda x: (x[0] != 'ensemble_average', x[0]))
        
        for model, price in sorted_preds:
            print(f"  {model:25s}: ${price:>10,.2f}")
        
        return result
    
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None



def predict_batch(samples):
    """
    Make batch predictions
    Args:
        samples: list of [weight, depth, table] arrays
    """
    try:
        payload = {"X": samples}
        
        response = requests.post(
            f"{API_BASE}/predict-batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        
        if response.status_code != 200:
            print(f"Batch prediction failed: {result.get('error', 'Unknown error')}")
            return None
        
        results_list = result.get('results', [])
        total = result.get('total_samples', 0)
        
        print(f"\nBATCH PREDICTIONS ({total} samples)")
        print("=" * 100)
        
        if not results_list:
            print(f"No results in response")
            return None
        
        for item in results_list:
            # Handle different key names
            sample_id = item.get('sample_id', item.get('sample', '?'))
            feat = item.get('features', {})
            preds = item.get('predictions', {})
            
            # Extract feature values with fallbacks
            weight = feat.get('weight_carats', feat.get('carat', '?'))
            depth = feat.get('depth_mm', feat.get('depth', '?'))
            table = feat.get('table_percent', feat.get('table', '?'))
            
            print(f"\n[Sample {sample_id}] Weight={weight:.2f}ct, Depth={depth:.1f}mm, Table={table:.1f}%")
            
            if preds:
                ensemble = preds.get('ensemble_average', '?')
                print(f"  Ensemble Avg: ${ensemble:,.2f}")
                
                # Show all models
                for model_name in sorted(preds.keys()):
                    if model_name != 'ensemble_average':
                        price = preds[model_name]
                        print(f"     {model_name:20s}: ${price:>10,.2f}")
        
        print("\n" + "=" * 100)
        return result
    
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None



if __name__ == "__main__":
    
    print("\n" + "="*100)
    print("ML REGRESSION API CLIENT")
    print("="*100)
    
    # Check API health
    print("\n🔍 Checking API...")
    health = health_check()
    if not health:
        print("Make sure Flask API is running: python ml_regression_api.py")
        exit(1)
    
    # Train models
    train_result = train_all_models()
    if not train_result:
        print("Training failed. Exiting.")
        exit(1)
    
    print("\n" + "="*100)
    print("TEST 1: SINGLE PREDICTION")
    print("="*100)
    predict_single(weight=0.5, depth=62.5, table=55.0)
    
    print("\n" + "="*100)
    print("TEST 2: BATCH PREDICTIONS (5 samples)")
    print("="*100)
    
    batch_samples = [
        [0.3, 61.0, 54.0],
        [0.5, 62.5, 55.0],
        [1.0, 63.0, 56.0],
        [1.5, 62.0, 57.0],
        [2.0, 61.5, 58.0]
    ]
    predict_batch(batch_samples)
    
    print("\nAll tests completed!\n")