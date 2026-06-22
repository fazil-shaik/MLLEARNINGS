"""
Data collection module with caching support.
Handles loading and caching of telco churn dataset.
"""
import pandas as pd
from functools import lru_cache
import os

class DataCollector:
    """Collects and caches telco churn data."""
    
    def __init__(self, data_path: str):
        """Initialize DataCollector with path to data."""
        self.data_path = data_path
        self._cache = None
    
    @property
    def data(self) -> pd.DataFrame:
        """Load data with caching."""
        if self._cache is None:
            self._cache = self._load_data()
        return self._cache
    
    def _load_data(self) -> pd.DataFrame:
        """Load telco churn dataset from CSV."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        return df
    
    def get_basic_info(self) -> dict:
        """Get basic information about the dataset."""
        df = self.data
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.to_dict()
        }
    
    def reset_cache(self):
        """Reset the data cache."""
        self._cache = None
