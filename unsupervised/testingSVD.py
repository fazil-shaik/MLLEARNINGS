import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

# Load sample image (china.jpg)
china = load_sample_image("china.jpg") / 255.0
# Convert to grayscale manually
gray = np.dot(china[...,:3], [0.2989, 0.5870, 0.1140])

# Perform SVD
U, S, Vt = np.linalg.svd(gray, full_matrices=False)

# Reconstruct with different ranks
ranks = [10, 50, 200]
fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(12, 4))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Original")
for i, k in enumerate(ranks, 1):
    img_recon = U[:,:k] @ np.diag(S[:k]) @ Vt[:k,:]
    axes[i].imshow(img_recon, cmap='gray')
    axes[i].set_title(f"k={k}")
plt.show()