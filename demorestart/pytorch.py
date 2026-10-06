#Basic implementation of nn
import math


#forward propogration

def simpleModel1(x1,x2,w1,w2,b):

    Z = x1*w1+x2*w2+b

    print("relu checking")

    reLu = max(0,Z)

    return reLu

def simpleModel2(x1,x2,w1,w2,b):

    Z = x1*w1+x2*w2+b

    print("relu checking")

    reLu = max(0,Z)

    return reLu


def finalOutput(x1,x2,w1,w2,b):
      Z = x1*w1+x2*w2+b
      print("relu checking")
      reLu = max(0,Z)
      return reLu
    
#adding sigmoid funciton-activation fucntion

def sigmoidForFInal(z):
    print("sigmoid function called....")
    return 1/(1+math.exp(-z))


#we got result after applying sigmoid with 88%


#adding softmax Activation function

def softmax(d,c,h):
     print("softmax fun called......")
     return max(d,c,h)


#backpropogation

import numpy as np


# Data
x = 2.0
y = 20.0


# Parameters
w1 = 3.0
b1 = 0.0

w2 = 2.0
b2 = 0.0


learning_rate = 0.01


for epoch in range(100):

    # FORWARD PROPAGATION

    h = w1 * x + b1

    y_pred = w2 * h + b2

    loss = (y - y_pred) ** 2


    # BACKPROPAGATION

    dL_dy_pred = 2 * (y_pred - y)

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

    w1 = w1 - learning_rate * dL_dw1
    b1 = b1 - learning_rate * dL_db1

    w2 = w2 - learning_rate * dL_dw2
    b2 = b2 - learning_rate * dL_db2


    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}: "
            f"Prediction={y_pred:.4f}, "
            f"Loss={loss:.4f}"
        )




print(simpleModel1(4,0.8,0.5,0.2,0.1))

print(simpleModel1(4,0.8,0.3,0.7,0.2))

print(finalOutput(2.26,1.56,0.6,0.4,0.1))

print(sigmoidForFInal(2.08))

print(softmax(0.88,0.12,0.56))

# import torch

# x = torch.tensor([[2.0]])

# y = torch.tensor([[20.0]])


# w = torch.tensor(
#     [[3.0]],
#     requires_grad=True
# )

# b = torch.tensor(
#     [[0.0]],
#     requires_grad=True
# )

# y_pred = x*w+b

# loss = (y-y_pred)**2


# print("Prediction:", y_pred.item())
# print("Loss:", loss.item())


# loss.backward()

# print("Gradient w:", w.grad.item())
# print("Gradient b:", b.grad.item())


# import torch
# import torch.nn as nnw

# model = nnw.Linear(10,1,bias=False)

# print(model)





# # making a network 

# import torch
# import torch.nn as nnw


# x = torch.tensor([[2.0]])
# y = torch.tensor([[2.0]])



# model = nnw.Linear(1,1)

# loss_fun = nnw.MSELoss()

# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr = 0.01
# )

# for epoch in range(100):

#     # Forward propagation
#     y_pred = model(x)

#     # Calculate loss
#     loss = loss_fun(y_pred, y)

#     # Clear old gradients
#     optimizer.zero_grad()

#     # Backpropagation
#     loss.backward()

#     # Update weight
#     optimizer.step()

#     if epoch % 10 == 0:
#         print(
#             f"Epoch {epoch}: "
#             f"Prediction={y_pred.item():.4f}, "
#             f"Loss={loss.item():.4f}"
#         )





#pytorch version

import torch
import torch.nn as nn


# Data
x = torch.tensor([[2.0]])
y = torch.tensor([[20.0]])


# Neural network
model = nn.Sequential(
    nn.Linear(1, 1),
    nn.Linear(1, 1)
)


# Loss function
loss_fn = nn.MSELoss()


# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# Training
for epoch in range(100):

    # FORWARD PROPAGATION

    y_pred = model(x)


    # LOSS

    loss = loss_fn(y_pred, y)


    # BACKPROPAGATION

    optimizer.zero_grad()

    loss.backward()


    # GRADIENT DESCENT

    optimizer.step()


    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}: "
            f"Prediction={y_pred.item():.4f}, "
            f"Loss={loss.item():.4f}"
        )