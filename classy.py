import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_bias_variance_learning_curve(model, X, y, param_range, param_name='max_depth'):
    """Plot learning curves to diagnose bias vs variance."""
    
    train_scores = []
    test_scores = []
    
    for param in param_range:
        # Set parameter (e.g., max_depth for tree, k for k-NN)
        model.set_params(**{param_name: param})
        
        # Calculate cross-validation scores
        train_scores_fold = []
        test_scores_fold = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            train_scores_fold.append(model.score(X_train_fold, y_train_fold))
            test_scores_fold.append(model.score(X_val_fold, y_val_fold))
        
        train_scores.append(np.mean(train_scores_fold))
        test_scores.append(np.mean(test_scores_fold))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores, 'b-', label='Training score', linewidth=2)
    plt.plot(param_range, test_scores, 'r-', label='Validation score', linewidth=2)
    plt.xlabel(param_name)
    plt.ylabel('Score (R² for regression, accuracy for classification)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotate regions
    plt.axvline(param_range[np.argmax(test_scores)], color='g', linestyle='--', 
                label='Optimal complexity')
    
    # Find where validation score starts decreasing
    max_idx = np.argmax(test_scores)
    if max_idx < len(param_range) - 1:
        plt.axvspan(param_range[0], param_range[max_idx], alpha=0.1, color='blue', 
                    label='High bias region')
    if max_idx > 0:
        plt.axvspan(param_range[max_idx], param_range[-1], alpha=0.1, color='red', 
                    label='High variance region')
    
    plt.title(f'Learning Curves: {param_name} vs Performance')
    plt.show()
    
    return train_scores, test_scores

# Usage
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)
model = DecisionTreeRegressor()
param_range = [1, 2, 3, 5, 8, 12, 16, 20]
train_scores, test_scores = plot_bias_variance_learning_curve(
    model, X, y, param_range, 'max_depth'
)