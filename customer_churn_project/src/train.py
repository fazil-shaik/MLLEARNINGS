"""
Model training module.
Handles model training and hyperparameter tuning.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle

class ModelTrainer:
    """Trains and tunes machine learning models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize trainer."""
        self.random_state = random_state
        self.model = None
        self.best_params = None
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """Train logistic regression model."""
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        return self.model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """Train random forest model."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        return self.model
    
    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """Train gradient boosting model."""
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=self.random_state,
            learning_rate=0.1
        )
        self.model.fit(X_train, y_train)
        return self.model
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_type: str = 'random_forest', cv: int = 5) -> dict:
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters
        """
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 5, 7]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        }
        
        models = {
            'random_forest': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=-1)
        }
        
        base_model = models.get(model_type)
        param_grid = param_grids.get(model_type)
        
        if base_model is None or param_grid is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        grid_search = GridSearchCV(base_model, param_grid, cv=cv, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.best_params
    
    def save_model(self, filepath: str):
        """Save trained model to pickle file."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from pickle file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return self.model
