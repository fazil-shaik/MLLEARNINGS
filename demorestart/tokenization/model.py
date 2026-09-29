import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head,
        block_size,
        dropout=0.1,
    ):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(
            n_embd,
            3 * n_embd
        )

        self.proj = nn.Linear(
            n_embd,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            ).view(
                1,
                1,
                block_size,
                block_size
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1
        )

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        attention = (
            q @ k.transpose(-2, -1)
        ) / (
            self.head_dim ** 0.5
        )

        attention = attention.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        attention = F.softmax(
            attention,
            dim=-1
        )

        attention = self.dropout(
            attention
        )

        y = attention @ v

        y = y.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )

        return self.proj(y)


class MLP(nn.Module):

    def __init__(self, n_embd, dropout=0.1):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                n_embd,
                4 * n_embd
            ),

            nn.GELU(),

            nn.Linear(
                4 * n_embd,
                n_embd
            ),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head,
        block_size,
        dropout=0.1
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)

        self.attention = CausalSelfAttention(
            n_embd,
            n_head,
            block_size,
            dropout
        )

        self.ln2 = nn.LayerNorm(n_embd)

        self.mlp = MLP(
            n_embd,
            dropout
        )

    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.mlp(
            self.ln2(x)
        )

        return x


class FazilLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd,
                    n_head,
                    block_size,
                    dropout
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(
            n_embd
        )

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        assert T <= self.block_size

        positions = torch.arange(
            0,
            T,
            device=idx.device
        )

        token_embeddings = (
            self.token_embedding(idx)
        )

        position_embeddings = (
            self.position_embedding(positions)
        )

        x = (
            token_embeddings
            + position_embeddings
        )

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(
                B * T,
                C
            )

            targets = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss