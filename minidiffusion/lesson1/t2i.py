import torch

# 0D tensor
scalar = torch.tensor(5)

# 1D tensor
vector = torch.tensor([1, 2, 3])

# 2D tensor
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

# 3D tensor representing an RGB image
image = torch.rand(3, 64, 64)

# Batch of 8 images
batch = torch.rand(8, 3, 64, 64)

print("Scalar:", scalar.shape)
print("Vector:", vector.size())


print("Matrix:", matrix.shape)
print("Image:", image.shape)
print("Batch:", batch.shape)




a = torch.tensor([2.0, 4.0, 6.0])
b = torch.tensor([1.0, 3.0, 5.0])

print("Addition:", a + b)
print("Multiplication:", a * b)
print("Mean:", a.mean())

image = torch.rand(3, 64, 64)

# Normalize image from [0, 1] to [-1, 1]
normalized = image * 2 - 1

print("Image shape:", normalized.shape)
print("Minimum:", normalized.min().item())
print("Maximum:", normalized.max().item())






import torch

# We want to learn y = 2x
x = torch.tensor([1., 2., 3., 4.])
y_true = 2 * x

# Start with an incorrect weight
w = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01

for epoch in range(100):
    # Forward pass
    y_pred = w * x

    # Calculate mean squared error
    loss = ((y_pred - y_true) ** 2).mean()

    # Calculate gradient
    loss.backward()

    # Update weight without tracking gradients
    with torch.no_grad():
        w -= learning_rate * w.grad

    # Clear old gradients
    w.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch+1}: \n"
            f"weight={w.item():.4f}, \n"
            f"loss={loss.item():.6f} \n"
        )

print("Final learned weight:", w.item())



x1 = torch.tensor([1,2,3,4])
x2 = torch.tensor([5,6,7,8])

imag1 = torch.rand(3,64,64)
imag2 = torch.rand(8,3,64,64)


result = imag1+imag2

print(result.shape)
print("mean of result",result.mean())
print(x1.shape)
print(x2.shape)
print(imag1.shape)
print(imag2.shape)
print(result.min())
print(result.max())
print("result mean  with item ",result.mean().item())


import torch

x = torch.tensor(3.0,requires_grad=True)

y = x**2


y.backward()


print("x:", x.item())
print("y:", y.item())
print("Gradient:", x.grad.item())




import torch

# Training data
x = torch.tensor([1., 2., 3., 4.])
y_true = torch.tensor([3., 6., 9., 12.])

# Trainable parameter
w = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.02

for epoch in range(100):

    # 1. Forward pass
    y_pred = w * x

    # 2. Calculate loss
    loss = ((y_pred - y_true) ** 2).mean()

    # 3. Calculate gradient
    loss.backward()

    # 4. Update weight
    with torch.no_grad():
        w -= learning_rate * w.grad

    # 5. Reset gradient
    w.grad.zero_()

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch}, "
            f"Weight: {w.item():.4f}, "
            f"Loss: {loss.item():.4f}"
        )

print("Final learned weight:", w.item())
