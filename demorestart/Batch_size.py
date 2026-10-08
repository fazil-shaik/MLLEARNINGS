import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nnw

X = torch.tensor([
    [10.0, 20.0],
    [30.0, 40.0],
    [50.0, 60.0],
    [70.0, 80.0],
    [90.0, 100.0],
    [110.0, 120.0]
])

y = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
])

print(f"shape of x is {X.shape}")

# Dataset
dataset = TensorDataset(X, y)

# Model
model = nnw.Sequential(
    nnw.Linear(2, 10),   # 2 input features
    nnw.ReLU(),
    nnw.Linear(10, 1)    # 1 output
)

# DataLoader
loader = DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=True
)

# Check batches
for X_batch, y_batch in loader:
    print("X batch:")
    print(X_batch)

    print("y batch:")
    print(y_batch)

    print("Shape:", X_batch.shape)
    print()

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.03
)

# Loss function
loss_fn = nnw.MSELoss()

# Training
for epoch in range(100):

    for X_batch, y_batch in loader:

        # 1. Clear gradients
        optimizer.zero_grad()

        # 2. Forward pass
        prediction = model(X_batch)

        # 3. Calculate loss
        loss = loss_fn(prediction, y_batch)

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")