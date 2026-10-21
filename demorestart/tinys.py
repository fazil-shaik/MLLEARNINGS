import numpy as np
import pandas as pd


with open("data/input.txt","r") as file:
    text = file.read()

print("Dataset characters:", len(text))
print(text[:500])


chars = sorted(set(text))

vocab_size = len(chars)

print("Vocabulary size:", vocab_size)
print(chars)


stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print(stoi["a"])
print(itos[stoi["a"]])


#tokenization

#encoding 

def encode(text):
    return [stoi[ch] for ch in text]

#decoding
def decode(token_ids):
    return "".join(itos[token_id] for token_id in token_ids)


sample = "Hello"

encoded = encode(sample)
decoded = decode(encoded)

print("Original:", sample)
print("Encoded:", encoded)
print("Decoded:", decoded)



#splitting training and val splits

import torch

data = torch.tensor(encode(text), dtype=torch.long)

split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]

print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))


#creating training batches

batch_size = 32
block_size = 8


def get_batch(split):
    source = train_data if split == "train" else val_data

    starts = torch.randint(
        len(source) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size] for i in starts
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1] for i in starts
    ])

    return x, y


x, y = get_batch("train")

print("Input shape:", x.shape)
print("Target shape:", y.shape)

print("Input IDs:", x[0])
print("Target IDs:", y[0])


#Building bigram model
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):
        logits = self.token_embedding(idx)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx

logits = torch.tensor([[2.0, 1.0, 0.1]])
target = torch.tensor([0])

loss = F.cross_entropy(logits, target)

print("Loss:", loss.item())

import torch

torch.manual_seed(1337)

model = BigramLanguageModel(vocab_size)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

max_iters = 10000
eval_interval = 300

@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for split in ["train", "val"]:
        losses = torch.zeros(100)

        for k in range(100):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {step}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    x, y = get_batch("train")

    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


start = torch.zeros((1, 1), dtype=torch.long)

generated_ids = model.generate(
    start,
    max_new_tokens=1000
)

generated_text = decode(generated_ids[0].tolist())

print("\nGenerated text:\n")
print(generated_text)