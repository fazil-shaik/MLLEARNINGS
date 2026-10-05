# First implementation of Neural Network

import numpy as np


# INPUT

x = 2


# HIDDEN LAYER PARAMETERS

w1 = 3.0
b1 = 0.0


# OUTPUT LAYER PARAMETERS

w2 = 2.0
b2 = 0.0


# ACTUAL VALUE

y = 20.0


# FORWARD PROPAGATION

print("=" * 30)
print("Forward Propagation")
print("=" * 30)


# Hidden neuron
h = w1 * x + b1


# Output neuron
y_predict = w2 * h + b2


# Loss
loss = (y - y_predict) ** 2


print("Hidden:", h)
print("Prediction:", y_predict)
print("Loss:", loss)


# BACKWARD PROPAGATION

print("=" * 30)
print("Backward Propagation")
print("=" * 30)


# Gradient of Loss with respect to Prediction
#
# dL/dŷ = 2(ŷ - y)

dL_dy_pred = 2 * (y_predict - y)


# Gradient of Prediction with respect to w2
#
# ŷ = w2 * h + b2
#
# dŷ/dw2 = h

dy_pred_dw2 = h


# Gradient of Loss with respect to w2
#
# dL/dw2 = dL/dŷ * dŷ/dw2

dL_dw2 = dL_dy_pred * dy_pred_dw2


# Hidden Layer

# dŷ/dh = w2
dy_pred_dh = w2


# dh/dw1 = x
dh_dw1 = x


# Chain Rule
#
# dL/dw1 =
# dL/dŷ * dŷ/dh * dh/dw1

dL_dw1 = (
    dL_dy_pred
    * dy_pred_dh
    * dh_dw1
)


print("dL/dw2:", dL_dw2)
print("dL/dw1:", dL_dw1)


# GRADIENT DESCENT

print("=" * 30)
print("Updated with Gradient Descent")
print("=" * 30)


learning_rate = 0.01


# Update hidden weight
w1 = w1 - learning_rate * dL_dw1


# Update output weight
w2 = w2 - learning_rate * dL_dw2


print("Updated w1:", w1)
print("Updated w2:", w2)


# TRAINING THROUGH EPOCHS

print("=" * 30)
print("Training Through Epochs")
print("=" * 30)


for epoch in range(100):

    # FORWARD PROPAGATION

    # Hidden neuron
    h = w1 * x + b1

    # Output neuron
    y_pred = w2 * h + b2

    # Loss
    loss = (y - y_pred) ** 2


    dL_dy_pred = 2 * (y_pred - y)


    # Output Weight

    dL_dw2 = dL_dy_pred * h


    dL_db2 = dL_dy_pred


    dL_dw1 = (
        dL_dy_pred
        * w2
        * x
    )


    dL_db1 = (
        dL_dy_pred
        * w2
    )


    # GRADIENT DESCENT

    # Update hidden weight
    w1 = w1 - learning_rate * dL_dw1

    # Update hidden bias
    b1 = b1 - learning_rate * dL_db1

    # Update output weight
    w2 = w2 - learning_rate * dL_dw2

    # Update output bias
    b2 = b2 - learning_rate * dL_db2


    # PRINT PROGRESS

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch}: "
            f"Prediction={y_pred:.4f}, "
            f"Loss={loss:.4f}"
        )


# FINAL PARAMETERS

print("=" * 30)
print("Final Parameters")
print("=" * 30)

print("w1:", w1)
print("b1:", b1)
print("w2:", w2)
print("b2:", b2)


# FINAL PREDICTION

h = w1 * x + b1
final_prediction = w2 * h + b2

final_loss = (y - final_prediction) ** 2


print("=" * 30)
print("Final Result")
print("=" * 30)

print("Hidden:", h)
print("Prediction:", final_prediction)
print("Actual:", y)
print("Loss:", final_loss)