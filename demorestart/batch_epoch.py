from torch.utils.data import DataLoader,TensorDataset
import torch


x = torch.randn(1000,10)
y = torch.randint(1,2,(1000, ))

dataset = TensorDataset(x,y)

loader = DataLoader(
    dataset=dataset,
    batch_size=100,
    shuffle=True
)

epochs = 5 

for epoch in range(epochs):
    for batch_X, batch_y in loader:
        # Training steps would happen here
        pass

    print(f"Epoch {epoch + 1} completed")



import torch
import torch.nn as nn

# Create a simple neural network
model = nn.Linear(10, 1)

# Define AdamW optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)

# Example input and target
X = torch.randn(32, 10)
y = torch.randn(32, 1)

# Define loss function
criterion = nn.MSELoss()

# Forward pass
predictions = model(X)
loss = criterion(predictions, y)

# Clear old gradients
optimizer.zero_grad()

# Calculate gradients
loss.backward()

# Update model weights
optimizer.step()

print("Loss:", loss.item())