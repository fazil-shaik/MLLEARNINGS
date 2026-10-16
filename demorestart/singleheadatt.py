import torch

import torch.nn as nn
import math

class selfAttentionSingleHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim,head_dim)
        self.key = nn.Linear(embed_dim,head_dim)
        self.value = nn.Linear(embed_dim,head_dim)

        self.head_dim = head_dim

    def forward(self,x):

        B,T,C = x.shape


        Q = self.query(x)
        K = self.key(x)
        V = self.key(x)


        scores = Q@K.transpose(-2,-1)

        scores = scores/math.sqrt(self.head_dim)


        weights = torch.softmax(scores,dim=1)

        ouptut = weights @ V

        return ouptut, weights 

torch.manual_seed(42)

x=torch.randn(2,3,8)


attention = selfAttentionSingleHead(
    embed_dim=8,
    head_dim=4
)

output, weigths = attention(x)

print("Input:", x.shape)
print("Output:", output.shape)
print("Weights:", weigths.shape)
print("Row sums:", weigths.sum(dim=-1))