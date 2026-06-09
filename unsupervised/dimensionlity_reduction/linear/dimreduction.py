import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset (real-world: replace with medical images)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, 
                                transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, 
                               transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define Autoencoder Architecture
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim=32):  # encoding_dim = bottleneck size
        super(Autoencoder, self).__init__()
        
        # Encoder: 784 pixels -> 32 numbers
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),  # 784 -> 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)  # 64 -> 32 (bottleneck!)
        )
        
        # Decoder: 32 numbers -> 784 pixels
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)  # (batch, 784)
        
        # Encode to bottleneck
        encoded = self.encoder(x)  # (batch, encoding_dim)
        
        # Decode back to image
        decoded = self.decoder(encoded)  # (batch, 784)
        
        # Reshape back to image
        decoded = decoded.view(x.size(0), 1, 28, 28)  # (batch, 1, 28, 28)
        
        return encoded, decoded

# Initialize model
encoding_dim = 32  # Compression ratio: 784/32 = 24.5x compression!
model = Autoencoder(encoding_dim=encoding_dim).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Minimize reconstruction error
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_autoencoder(model, train_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            encoded, decoded = model(data)
            loss = criterion(decoded, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# Train the model
print("Training autoencoder...")
train_autoencoder(model, train_loader, epochs=15)

# Visualize results
def visualize_reconstruction(model, test_loader, num_images=5):
    model.eval()
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        encoded, decoded = model(images)
    
    # Plot results
    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(decoded[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')
    
    plt.suptitle(f"Autoencoder Compression: {784} → {encoding_dim} → {784}", 
                 fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Also print the bottleneck representation
    print(f"\nBottleneck representation for first image (dimension {encoding_dim}):")
    print(encoded[0].cpu().numpy()[:10], "...")  # Show first 10 numbers

visualize_reconstruction(model, test_loader)

# REAL-WORLD USE: Extract compressed features for downstream tasks
def extract_compressed_features(model, dataloader):
    """Use autoencoder as a feature extractor"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            encoded, _ = model(data)  # Only need the encoded features
            features.append(encoded.cpu().numpy())
            labels.append(target.numpy())
    
    return np.concatenate(features), np.concatenate(labels)

# Extract compressed features (32-dim instead of 784-dim)
compressed_features, labels = extract_compressed_features(model, test_loader)
print(f"\nOriginal image dimension: 784")
print(f"Compressed feature dimension: {compressed_features.shape[1]}")
print(f"Compression ratio: {784/compressed_features.shape[1]:.1f}x")

# Bonus: Use compressed features for classification
from sklearn.cluster import KMeans

# Show that compressed features preserve meaningful structure
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(compressed_features)

# Visualize compressed features in 2D (using PCA on the bottleneck)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
features_2d = pca.fit_transform(compressed_features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                      c=labels, cmap='tab10', alpha=0.6, s=10)
plt.colorbar(scatter)
plt.title("Autoencoder Compressed Features (32D → 2D PCA)\nColors = True Digit Labels")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()