import tensorflow as tf
print("TensorFlow version:", tf.__version__)

minst = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = minst.load_data()

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) 

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc) 
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)
print("Predictions for the first test image:", predictions[0])

