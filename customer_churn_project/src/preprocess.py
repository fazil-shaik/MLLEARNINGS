"""
Data preprocessing module.
Handles data cleaning, feature engineering, and scaling.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Handles data preprocessing and feature engineering."""
    
    def __init__(self, random_state: int = 42):
        """Initialize preprocessor."""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = 'Churn') -> tuple:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            
        Returns:
            Tuple of (X_transformed, y)
        """
        self.target_column = target_column
        
        # Separate features and target
        y = df[target_column].copy()
        X = df.drop(target_column, axis=1).copy()
        
        # Encode categorical variables
        X = self._encode_categorical(X)
        
        # Handle missing values
        X = self._handle_missing(X)
        
        # Scale numerical features
        X = self._scale_features(X, fit=True)
        
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def transform(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        X = df.drop(target_column or self.target_column, axis=1, errors='ignore').copy()
        
        X = self._encode_categorical(X, fit=False)
        X = self._handle_missing(X)
        X = self._scale_features(X, fit=False)
        
        return X
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        df = df.copy()
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
        """Split data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)
