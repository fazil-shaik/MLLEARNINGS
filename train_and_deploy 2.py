# train_and_deploy.py
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'your_model.pkl')
print("Model saved successfully!")

# Test prediction
sample = X_test[0:1]
prediction = model.predict(sample)
probability = model.predict_proba(sample)
print(f"Sample prediction: {prediction[0]}")
print(f"Sample probability: {probability[0]}")