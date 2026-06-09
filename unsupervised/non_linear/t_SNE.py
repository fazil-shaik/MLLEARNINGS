import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data 
y = digits.target  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(n_components=2, 
            perplexity=30, 
            learning_rate='auto',
            max_iter=1000,
            random_state=42)

X_tsne = tsne.fit_transform(X_scaled)  # Shape: (1797, 2)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                      c=y, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Digit Class')
plt.title('t-SNE Visualization of Handwritten Digits (64D → 2D)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_tsne.shape[1]}")