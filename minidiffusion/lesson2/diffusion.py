import torch

noise = torch.randn(10)

print("Mean:", noise.mean().item())
print("Std:", noise.std().item())
print("Minimum:", noise.min().item())
print("Maximum:", noise.max().item())


import torch

T = 1000

betas = torch.linspace(0.0001, 0.02, T)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(alphas, dim=0)

print("First alpha bar:", alpha_bars[0].item())
print("Middle alpha bar:", alpha_bars[499].item())
print("Last alpha bar:", alpha_bars[-1].item())


import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load image
image = Image.open("sample.jpg").convert("RGB")
image = image.resize((64, 64))

# Convert image to tensor: HWC -> CHW
array = np.asarray(image).copy()

x0 = torch.from_numpy(array).float() / 255.0
x0 = x0.permute(2, 0, 1)

# Normalize image to [-1, 1]
x0 = x0 * 2 - 1

# Diffusion settings
T = 1000
betas = torch.linspace(0.0001, 0.02, T)

alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# Add noise at a specific timestep
def add_noise(image, t):
    epsilon = torch.randn_like(image)

    alpha_bar = alpha_bars[t]

    noisy = (
        torch.sqrt(alpha_bar) * image
        + torch.sqrt(1 - alpha_bar) * epsilon
    )

    return noisy

# Convert tensor back to displayable image
def to_display(tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    return tensor.permute(1, 2, 0).numpy()

timesteps = [0, 100, 300, 600, 999]

plt.figure(figsize=(12, 3))

for i, t in enumerate(timesteps):
    noisy_image = add_noise(x0, t)

    plt.subplot(1, len(timesteps), i + 1)
    plt.imshow(to_display(noisy_image))
    plt.title(f"t = {t}")
    plt.axis("off")

plt.tight_layout()
plt.show()
