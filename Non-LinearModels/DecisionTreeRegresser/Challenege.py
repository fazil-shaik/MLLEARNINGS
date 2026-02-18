import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

np.random.seed(42)

n = 300

X1 = np.random.rand(n) * 10
X2 = np.random.rand(n) * 5
X3 = np.random.rand(n) * 7
X_noise1 = np.random.randn(n)
X_noise2 = np.random.randn(n)
X_noise3 = np.random.randn(n)
X = np.column_stack([X1, X2, X3,X_noise1, X_noise2,X_noise3])

y = 5000*X1 + 10000*(X2**2) - 1500*(X3**3) + np.random.randn(n)*10000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

importances = dt.feature_importances_
print(importances)

plot_tree(
    dt,feature_names=X,rounded=True,max_depth=3
)
plt.show()

feature_names = ["Size", "Rooms", "distance","Noise1", "Noise2","Noise3"]

plt.figure()
plt.bar(feature_names, importances)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance - Decision Tree")
plt.show()

# indices = np.argsort(importances)[::-1]

# plt.figure()
# plt.bar(np.array(feature_names)[indices], importances[indices])
# plt.xlabel("Features")
# plt.ylabel("Importance Score")
# plt.title("Sorted Feature Importance")
# plt.show()
