import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, head_dim):
        super().__init__()

        # Trainable Query, Key, and Value projections
        self.query = nn.Linear(d_model, head_dim, bias=False)
        self.key = nn.Linear(d_model, head_dim, bias=False)
        self.value = nn.Linear(d_model, head_dim, bias=False)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        B, T, C = x.shape

        # 1. Project input into Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 2. Calculate scaled attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(K.size(-1))

        # 3. Prevent attention to future tokens
        future_mask = torch.triu(
            torch.ones(
                T, T,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            future_mask,
            float("-inf")
        )

        # 4. Convert scores to probabilities
        weights = F.softmax(scores, dim=-1)

        # 5. Combine Value vectors
        output = weights @ V

        return output, weights


# TEST 1: Shapes and causal masking

torch.manual_seed(42)

B = 2
T = 4
C = 8
D = 4

x = torch.randn(B, T, C)

attention = CausalSelfAttention(
    d_model=C,
    head_dim=D
)

output, weights = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)

assert output.shape == (B, T, D)
assert weights.shape == (B, T, T)

# Attention probabilities should sum to 1
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(B, T),
    atol=1e-6
)

# Future-token attention should be zero
assert torch.all(
    torch.triu(weights, diagonal=1) == 0
)

print("Shape and causal-mask tests passed!")


# TEST 2: Backpropagation

loss = output.square().mean()

attention.zero_grad()
loss.backward()

print("\nLoss:", loss.item())

print(
    "Query weight gradient exists:",
    attention.query.weight.grad is not None
)

print(
    "Key weight gradient exists:",
    attention.key.weight.grad is not None
)

print(
    "Value weight gradient exists:",
    attention.value.weight.grad is not None
)

assert attention.query.weight.grad is not None
assert attention.key.weight.grad is not None
assert attention.value.weight.grad is not None

print("Gradient tests passed!")
