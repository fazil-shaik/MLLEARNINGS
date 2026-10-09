# from cv2 import transform
# import torch
# import torch.nn as nn

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader



# # 1. Transform

# transform = transforms.ToTensor()


# # 2. Dataset

# train_dataset = datasets.MNIST(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transform
# )


# # 3. DataLoader

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=64,
#     shuffle=True
# )


# # 4. Model

# model = nn.Sequential(
#     nn.Linear(784, 128),
#     nn.ReLU(),
#     nn.Linear(128, 10)
# )


# # 5. Loss

# loss_fn = nn.CrossEntropyLoss()


# # 6. Optimizer

# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=0.001
# )


# for epoch in range(5):

#     for images, labels in train_loader:

#         # Flatten image
#         images = images.view(images.size(0), -1)

#         # Clear gradients
#         optimizer.zero_grad()

#         # Forward
#         outputs = model(images)

#         # Loss
#         loss = loss_fn(outputs, labels)

#         # Backpropagation
#         loss.backward()

#         # Update
#         optimizer.step()

#     if (epoch + 1) % 1 == 0:
#         print(f"Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}")






import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Transform
transform = transforms.ToTensor()

# 3. Training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# 4. Testing dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 5. DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# 6. Model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

# 7. Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 8. Training loop
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # 9. Evaluate on unseen test images
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Test Accuracy: {accuracy:.2f}%"
    )
import matplotlib.pyplot as plt

model.eval()

image, actual_label = test_dataset[0]

with torch.no_grad():
    image_input = image.view(1, 784).to(device)
    output = model(image_input)
    predicted_label = output.argmax(dim=1).item()

plt.imshow(image.squeeze(), cmap="gray")
plt.title(
    f"Actual: {actual_label}, "
    f"Predicted: {predicted_label}"
)
plt.axis("off")
plt.show()

print("Actual digit:", actual_label)
print("Predicted digit:", predicted_label)