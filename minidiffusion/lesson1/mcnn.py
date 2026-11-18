import torch
import torch.nn as nn

class MyFirstCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # RGB image: 3 × 64 × 64
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            # 16 × 64 × 64 -> 16 × 32 × 32
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            # 32 × 32 × 32 -> 32 × 16 × 16
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # 64 × 16 × 16 -> 64 × 8 × 8
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        print("After convolution:", x.shape)

        x = self.classifier(x)
        return x


model = MyFirstCNN()

# Batch of 8 RGB images
images = torch.rand(8, 3, 64, 64)

outputs = model(images)

print("Final output:", outputs.shape)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters())
)

test_input = torch.rand(1, 3, 64, 64)
test_output = model(test_input)

print("Test input shape:", test_input.shape)
print("Test output shape:", test_output.shape)




import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

x = torch.rand(8, 3, 64, 64)

print("Input:", x.shape)

for i, layer in enumerate(model):
    x = layer(x)
    print(
        f"Layer {i}: "
        f"{layer.__class__.__name__} "
        f"-> {tuple(x.shape)}"
    )
