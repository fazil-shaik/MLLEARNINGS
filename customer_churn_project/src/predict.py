"""
Inference service module with caching.
Handles model predictions with result caching.
"""
import pandas as pd
import numpy as np
from functools import lru_cache
import pickle
import hashlib

class PredictionService:
    """Handles model predictions with caching support."""
    
    def __init__(self, model_path: str):
        """Initialize prediction service with trained model."""
        self.model = self._load_model(model_path)
        self._prediction_cache = {}
    
    def _load_model(self, model_path: str):
        """Load trained model from pickle file."""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {model_path}")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict(self, X: pd.DataFrame, use_cache: bool = True) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            use_cache: Whether to use cached results
            
        Returns:
            Predicted labels
        """
        cache_key = self._get_cache_key(X) if use_cache else None
        
        if use_cache and cache_key in self._prediction_cache:
            print(f"Using cached predictions for {len(X)} samples")
            return self._prediction_cache[cache_key]
        
        predictions = self.model.predict(X)
        
        if use_cache:
            self._prediction_cache[cache_key] = predictions
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame, use_cache: bool = True) -> np.ndarray:
        """
        Get prediction probabilities on new data.
        
        Args:
            X: Input features
            use_cache: Whether to use cached results
            
        Returns:
            Prediction probabilities
        """
        cache_key = f"proba_{self._get_cache_key(X)}" if use_cache else None
        
        if use_cache and cache_key in self._prediction_cache:
            print(f"Using cached probabilities for {len(X)} samples")
            return self._prediction_cache[cache_key]
        
        probabilities = self.model.predict_proba(X)
        
        if use_cache:
            self._prediction_cache[cache_key] = probabilities
        
        return probabilities
    
    def _get_cache_key(self, X: pd.DataFrame) -> str:
        """Generate cache key for dataframe."""
        data_hash = hashlib.md5(pd.util.hash_pandas_object(X, index=True).values).hexdigest()
        return data_hash
    
    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get predictions with confidence scores.
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Get max probability as confidence
        confidence = np.max(probabilities, axis=1)
        
        result_df = pd.DataFrame({
            'prediction': predictions,
            'no_churn_prob': probabilities[:, 0],
            'churn_prob': probabilities[:, 1],
            'confidence': confidence
        })
        
        return result_df
    
    def clear_cache(self):
        """Clear prediction cache."""
        self._prediction_cache = {}
        print("Prediction cache cleared")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'cached_predictions': len(self._prediction_cache),
            'cache_memory_usage': sum(
                p.nbytes if isinstance(p, np.ndarray) else 0 
                for p in self._prediction_cache.values()
            ) / (1024 * 1024)  # Convert to MB
        }
