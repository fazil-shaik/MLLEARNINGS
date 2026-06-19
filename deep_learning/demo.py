import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import pandas as pd

np.random.seed(42)
num_samples = 100000

noise = np.random.normal(0, 0.1, num_samples)

# Create a DataFrame

df = pd.DataFrame({
    "soil_moisture": np.random.uniform(20, 80, num_samples)+noise, # Percentage
    "temperature": np.random.uniform(15, 35, num_samples), # Celsius
    "humidity": np.random.uniform(40, 95, num_samples), # Percentage
    "sunlight_hours": np.random.uniform(4, 12, num_samples), # Hours
    "yield": np.random.uniform(50, 200, num_samples), # Example yield value
    "need_water": np.random.choice([0, 1], num_samples)
})

print(df.head())

X = df[["soil_moisture","temperature","sunlight_hours"]]
y = df['need_water']


X_min = X.min()
X_max = X.max()
X_scaled = (X-X_min)/(X_max-X_min+1e-8)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=20,random_state=42)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(
    X_train.values,y_train.values,
    validation_data = (X_test.values,y_test.values),
    epochs=5,
    batch_size=4
)

print(history.history)
print("Training complete.")
print("Evaluating on training data...")
train_loss, train_acc = model.evaluate(X_train.values, y_train.values)
print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

print("Evaluating on test data...")
test_loss, test_acc = model.evaluate(X_test.values, y_test.values)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()