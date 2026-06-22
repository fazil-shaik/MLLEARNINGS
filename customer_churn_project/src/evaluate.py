"""
Model evaluation module.
Handles model evaluation metrics and visualization.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Evaluates model performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = None
        self.confusion_mat = None
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_pred_proba: np.ndarray = None) -> dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }
        
        if y_pred_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        self.confusion_mat = confusion_matrix(y_true, y_pred)
        
        return self.metrics
    
    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Print detailed classification report."""
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
        
        if self.metrics:
            print("\nKEY METRICS:")
            for metric, value in self.metrics.items():
                print(f"{metric.upper()}: {value:.4f}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             figsize: tuple = (8, 6)):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      figsize: tuple = (8, 6)):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        return plt
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get metrics as a DataFrame for easy viewing."""
        if self.metrics is None:
            return pd.DataFrame()
        return pd.DataFrame([self.metrics]).T.rename(columns={0: 'Score'})
