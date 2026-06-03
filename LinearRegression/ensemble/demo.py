import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from collections import Counter

# Create synthetic dataset (e.g., customer churn prediction)
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=8,
    n_redundant=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# IMPLEMENT BAGGING FROM SCRATCH

class BaggingClassifier:
    def __init__(self, base_model, n_estimators=10, sample_size=0.8):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.models = []
    
    def bootstrap_sample(self, X, y):
        """Create one bootstrap sample with replacement"""
        n_samples = int(len(X) * self.sample_size)
        indices = np.random.choice(len(X), n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """Train multiple models on bootstrap samples"""
        self.models = []
        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)
            
            # Train model on this sample
            model = clone_model(self.base_model)
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)
            
            print(f"Trained model {i+1}/{self.n_estimators}")
    
    def predict(self, X):
        """Aggregate predictions: majority vote"""
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Majority vote for each sample
        final_predictions = []
        for sample_idx in range(predictions.shape[1]):
            votes = predictions[:, sample_idx]
            majority_vote = Counter(votes).most_common(1)[0][0]
            final_predictions.append(majority_vote)
        
        return np.array(final_predictions)

def clone_model(model):
    """Helper to create fresh copy of model"""
    from sklearn.base import clone
    return clone(model)

# Compare single tree vs bagged trees
single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
single_tree.fit(X_train, y_train)
single_pred = single_tree.predict(X_test)
single_accuracy = accuracy_score(y_test, single_pred)

# Bagging with 50 trees
bagging_model = BaggingClassifier(
    base_model=DecisionTreeClassifier(max_depth=5),
    n_estimators=50,
    sample_size=0.8
)
bagging_model.fit(X_train, y_train)
bagging_pred = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_pred)

print(f"\n{'='*50}")
print(f"Single Decision Tree Accuracy: {single_accuracy:.3f}")
print(f"Bagging (50 trees) Accuracy: {bagging_accuracy:.3f}")
print(f"Improvement: {(bagging_accuracy - single_accuracy)*100:.1f}%")
print(f"{'='*50}")


# USING SKLEARN'S BAGGING (Production Ready)

from sklearn.ensemble import BaggingClassifier

sklearn_bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=50,
    bootstrap=True,  # Use bootstrap sampling
    random_state=42
)

sklearn_bagging.fit(X_train, y_train)
sklearn_pred = sklearn_bagging.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_pred)

print(f"\nScikit-learn Bagging Accuracy: {sklearn_accuracy:.3f}")


# DEMONSTRATE VARIANCE REDUCTION

def show_variance_reduction():
    """Train same model on different data splits to show stability"""
    
    print("\n" + "="*50)
    print("VARIANCE REDUCTION DEMONSTRATION")
    print("="*50)
    
    # Take 5 different random 80% samples
    single_tree_accuracies = []
    bagging_accuracies = []
    
    for run in range(5):
        # Different random split each time
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=run
        )
        
        # Single tree
        tree = DecisionTreeClassifier(max_depth=5)
        tree.fit(X_tr, y_tr)
        single_acc = accuracy_score(y_te, tree.predict(X_te))
        single_tree_accuracies.append(single_acc)
        
        # Bagging
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=30,
            random_state=42
        )
        bag.fit(X_tr, y_tr)
        bag_acc = accuracy_score(y_te, bag.predict(X_te))
        bagging_accuracies.append(bag_acc)
    
    print(f"Single Tree - Mean: {np.mean(single_tree_accuracies):.3f}, "
          f"Std Dev: {np.std(single_tree_accuracies):.4f}")
    print(f"Bagging     - Mean: {np.mean(bagging_accuracies):.3f}, "
          f"Std Dev: {np.std(bagging_accuracies):.4f}")
    print(f"✓ Bagging reduces variance by "
          f"{np.std(single_tree_accuracies)/np.std(bagging_accuracies):.1f}x")

show_variance_reduction()