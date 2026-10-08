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

dataset = TensorDataset(X,y)

model = nnw.Sequential(
    nnw.Linear(2, 10),
    nnw.ReLU(),
    nnw.Linear(10, 1)
)
# prediction = model(X)

loader = DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=True
)


for X_batch,y_batch in loader:
    print("X batch:")
    print(X_batch)

    print("y batch:")
    print(y_batch)

    print("Shape:", X_batch.shape)
    print()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

loss_fn = nnw.MSELoss()


for epoch in range(100):

    for X_batch, y_batch in loader:

        # 1. Clear gradients
        optimizer.zero_grad()

        # 2. Forward
        prediction = model(X_batch)

        # 3. Loss
        loss = loss_fn(prediction, y_batch)

        # 4. Backpropagation
        loss.backward()

        # 5. Update
        optimizer.step()

