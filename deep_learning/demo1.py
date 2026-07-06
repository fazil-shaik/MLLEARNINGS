import numpy as np
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import pandas as pd


df = pd.DataFrame({
    "soil_moisture": np.random.rand(1000),
    "temperature_c": np.random.randint(15, 35, size=1000),
    "sunlight_hours": np.random.randint(1,10, size=1000),
    "needs_water": np.random.randint(0, 2, size=1000),
})

print(df.head())

X = df.drop('needs_water', axis=1)
y = df['needs_water']

X_min = X.min()
X_max = X.max()
X_scaled = (X-X_min)/(X_max-X_min+1e-8)

# print("======="*10)
# print("X_scaled values are :")
# print(X_scaled)



X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=25,random_state=42)

 
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], ) ),
    layers.Dense(8,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

X_train_np = X_train.values.astype("float32")
X_test_np  = X_test.values.astype("float32")
y_train_np = y_train.values.astype("float32")
y_test_np  = y_test.values.astype("float32")

history_full = model.fit(
    X_train_np, y_train_np,
    epochs=100, batch_size=len(X_train_np), verbose=1
)

history = model.fit(
    X_train_np, y_train_np,
    validation_data=(X_test_np, y_test_np),
    epochs=100, batch_size=1, verbose=1
)

history_minibatch = model.fit(
    X_train_np, y_train_np,
    epochs=100, batch_size=100, verbose=1
)


