import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(100, 100, dtype=torch.bool))
        )

        self.head_dim = head_dim

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            ~self.mask[:T, :T],
            float("-inf")
        )

        weights = torch.softmax(scores, dim=-1)

        output = weights @ V

        return output, weights


torch.manual_seed(42)

B, T, C = 2, 4, 8

x = torch.randn(B, T, C)

attention = CausalSelfAttention(
    embed_dim=C,
    head_dim=4
)

output, weights = attention(x)

print("Input:", x.shape)
print("Output:", output.shape)
print("Weights:", weights.shape)

print(
    "Attention row sums:",
    weights.sum(dim=-1)
)

print(
    "Future attention weights:",
    weights[:, 0, 1:]
)

loss = output.square().mean()
loss.backward()

print(
    "Query gradient exists:",
    attention.query.weight.grad is not None
)

print(
    "Key gradient exists:",
    attention.key.weight.grad is not None
)

print(
    "Value gradient exists:",
    attention.value.weight.grad is not None
)