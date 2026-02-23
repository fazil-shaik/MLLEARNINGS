import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
np.random.seed(42)

X = np.random.uniform(-3, 3, size=100)
y = 2*X**3 +0.5*X**2 + X + 3 + np.random.normal(0,3,size=100)


plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Nonlinear Relationship')
plt.show()

#Decision Tree Regressor
X = X.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Decisionmodel = DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)

Decisionmodel.fit(X_train, y_train)


y_pred = Decisionmodel.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Squared error on training set:", mean_squared_error(y_train, Decisionmodel.predict(X_train)))
print("R2 Score:", Decisionmodel.score(X_test, y_test))
print("R2 Score on training set:", Decisionmodel.score(X_train, y_train))
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
y_plot = Decisionmodel.predict(X_plot)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data')
plt.plot(X_plot, y_plot, color='red', label='Decision Tree Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()

plot_tree(Decisionmodel,filled=True,feature_names=['X'], rounded=True,max_depth=3)
plt.show()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(Decisionmodel, X, y, cv=5, scoring='r2')
print(scores.mean())