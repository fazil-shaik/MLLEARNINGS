import torch.nn as nn
import torch


pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.rand(1, 16, 64, 64)
output = pool(x)

print(output.shape)
# torch.Size([1, 16, 32, 32])