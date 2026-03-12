import math

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
SEED = 23

X, y = load_digits(return_X_y=True)

train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = SEED)

gbc = GradientBoostingClassifier(n_estimators=300,
                                 learning_rate=0.05,
                                 random_state=100,
                                 max_features=5 )
                                 
gbc.fit(train_X, train_y)

pred_y = gbc.predict(test_X)

acc = accuracy_score(test_y, pred_y)
print("Gradient Boosting Classifier accuracy is : {:.2f}".format(acc))


# plt.figure(figsize=(10,7))
# plt.scatter(train_X[:, 0],train_y,label='Orginal data')
# plt.plot(pred_y,label='Predicted value')
# plt.show()


num_all_features = train_X.shape[1] 

# Let's only plot the first 6 features so it's readable
features_to_plot = 6 
cols = 2
rows = math.ceil(features_to_plot / cols)

fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
axes = axes.flatten()

for i in range(features_to_plot):
    # 2. FIX: Access NumPy columns using slicing [:, i]
    axes[i].scatter(train_X[:, i], train_y, alpha=0.5, color='teal')
    axes[i].plot(pred_y,alpha=0.2,color='red')
    axes[i].set_title(f'Pixel {i} vs Digit Class')
    axes[i].set_xlabel(f'Pixel {i} Intensity')
    axes[i].set_ylabel('Digit (0-9)')

plt.tight_layout()
plt.show()