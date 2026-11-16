import torch

import torch.nn as nn


torch.manual_seed(42)

X = torch.rand(1000,2)

y = 5*X[:, 0:1] + 4*X[:, 1:2] + 2


model = nn.Sequential(
    nn.Linear(2,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,1),
)


criteria = nn.MSELoss()



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(600):
    predictions = model(X)

    loss = criteria(predictions,y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch+1} | "
            f"Loss: {loss.item():.6f}"
        )

# Test the network
model.eval()

test_input = torch.tensor([[0.2, 0.6]])

with torch.no_grad():
    prediction = model(test_input)

print("Predicted:", prediction.item())
print("Expected:", 5 * 0.2 + 4 * 0.6 + 2)




