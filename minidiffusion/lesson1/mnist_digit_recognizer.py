"""Train a PyTorch CNN to classify handwritten digits (0-9) using MNIST.

Examples:
  python mnist_digit_recognizer.py --epochs 5
  python mnist_digit_recognizer.py --predict my_digit.png
  python mnist_digit_recognizer.py --epochs 5 --predict my_digit.png
"""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 x 28 -> 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 x 14 -> 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),  # One score for each digit: 0 through 9
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon Macs
    return torch.device("cpu")


def mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_loaders(batch_size):
    train_set = datasets.MNIST(
        root="data", train=True, download=True, transform=mnist_transform()
    )
    test_set = datasets.MNIST(
        root="data", train=False, download=True, transform=mnist_transform()
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def preprocess_custom_image(path):
    """Convert a handwritten digit on light OR dark background into MNIST-like input."""
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)

    # MNIST uses a light-colored digit against a dark background.
    width, height = img.size
    pixels = img.load()
    border_values = (
        [pixels[x, 0] for x in range(width)]
        + [pixels[x, height - 1] for x in range(width)]
        + [pixels[0, y] for y in range(height)]
        + [pixels[width - 1, y] for y in range(height)]
    )
    if sum(border_values) / len(border_values) > 127:
        img = ImageOps.invert(img)

    # Crop the digit, ignoring faint background noise.
    mask = img.point(lambda p: 255 if p > 35 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("No visible digit found in the image.")
    img = img.crop(bbox)

    # Resize while preserving aspect ratio; place on 28 x 28 canvas.
    w, h = img.size
    scale = 20 / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(img, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))

    tensor = mnist_transform()(canvas).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_image(model, path, device):
    model.eval()
    image = preprocess_custom_image(path).to(device)
    logits = model(image)
    probs = torch.softmax(logits, dim=1)[0]
    digit = int(probs.argmax().item())
    confidence = float(probs[digit].item() * 100)
    print(f"Image: {path}")
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {confidence:.2f}%")
    return digit


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST CNN digit recognizer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", type=Path, default=Path("digit_cnn.pth"))
    parser.add_argument("--predict", type=Path, help="Predict one digit image")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = choose_device()
    print(f"Using device: {device}")
    model = DigitCNN().to(device)

    if args.predict and args.model.exists():
        # In prediction mode, reuse an existing trained model without retraining.
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    else:
        if args.predict:
            print(f"No trained model at {args.model}. Training a new one first...")
        train_loader, test_loader = get_loaders(args.batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        best_accuracy = 0.0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device
            )
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}% | "
                f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%"
            )
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(model.state_dict(), args.model)
                print(f"  Saved best model to {args.model}")

        if args.model.exists():
            model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
            print(f"Best test accuracy: {best_accuracy:.2f}%")
        else:
            raise RuntimeError("No model saved. Ensure --epochs is at least 1.")

        # Try a sample from the MNIST test set.
        sample, true_label = test_loader.dataset[0]
        logits = model(sample.unsqueeze(0).to(device))
        predicted = int(logits.argmax(dim=1).item())
        print(f"MNIST example: predicted={predicted}, actual={true_label}")

    if args.predict:
        predict_image(model, args.predict, device)


if __name__ == "__main__":
    main()
