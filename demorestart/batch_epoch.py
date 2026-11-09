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





import numpy as np

# Step 1: Represent three tokens
# I = [1, 0]
# love = [0, 1]
# ML = [1, 1]

X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

# Step 2: For this example, use identity projections.
# Therefore Q = K = V = X.

Q = X
K = X
V = X

# Step 3: Calculate attention scores
d_k = K.shape[1]

scores = Q @ K.T

print("Raw attention scores:")
print(scores)

# Step 4: Scale the scores
scaled_scores = scores / np.sqrt(d_k)

print("\nScaled attention scores:")
print(np.round(scaled_scores, 3))

# Step 5: Calculate softmax
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)

print("\nAttention weights:")
print(np.round(attention_weights, 3))

# Step 6: Calculate attention output
output = attention_weights @ V

print("\nAttention output:")
print(np.round(output, 3))

# Step 7: Display the result for the first token
print("\nOutput for token 'I':")
print(np.round(output[0], 3))