import torch
import torch.nn as nn

conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1
)

image = torch.rand(1, 3, 64, 64)

output = conv(image)

print(output.shape)