from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score,mean_absolute_error,accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

np.random.seed(42)

data_size = 400


data = {
    'transaction_amount':np.random.uniform(5,5000,data_size),
    'is_international':np.random.choice([0,1],data_size,p=[0.8,0.2]),
    'hour_of_day': np.random.randint(0, 24, data_size),
    'attempts_last_hour': np.random.randint(1, 10, data_size)
}

df = pd.DataFrame(data=data)


df['is_fraud'] = (
    (df['transaction_amount'] > 3000) & (df['is_international'] == 1) | 
    (df['attempts_last_hour'] > 7)
).astype(int)

# Add a bit of chaos (Noise)
# Flip the fraud status for 5% of the rows randomly
noise_mask = np.random.random(data_size) < 0.05
df.loc[noise_mask, 'is_fraud'] = 1 - df.loc[noise_mask, 'is_fraud']

# 2. SETUP FEATURES (X) AND TARGET (y)
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# 3. SPLIT THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. INITIALIZE AND TRAIN
# We'll use 100 trees (n_estimators)
model = RandomForestClassifier(n_estimators=400,max_depth=5,min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# 5. PREDICT AND EVALUATE
predictions = model.predict(X_test)

print("--- Model Results ---")
print(f"Accuracy Score: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 6. SEE WHICH FEATURES MATTERED MOST
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(importances.sort_values(ascending=False))

# 1. Predict on Training data
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

# 2. Predict on Testing data
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# 3. Calculate the "Gap"
gap = train_accuracy - test_accuracy
print(f"Accuracy Gap: {gap * 100:.2f}%")
scores = cross_val_score(model, X, y, cv=5)

print(f"All Scores: {scores}")
print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores.std():.4f}")

# Generate the matrix
cm = confusion_matrix(y_test, predictions)

# Visualize it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fraud'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Fraud Detection: Confusion Matrix")
plt.show()