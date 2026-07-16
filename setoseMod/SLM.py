#converting text to numbers 
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

encoded = [char_to_idx[ch] for ch in text]

print(encoded[:20])


#training the data
import torch

block_size = 8

X = []
Y = []

for i in range(len(encoded) - block_size):
    x = encoded[i:i + block_size]
    y = encoded[i + 1:i + block_size + 1]

    X.append(x)
    Y.append(y)

X = torch.tensor(X)
Y = torch.tensor(Y)

print(X.shape)
print(Y.shape)


#neural network loop train implementation
import torch.nn as nn

class TinySLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim=64):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.fc = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        logits = self.fc(x)

        return logits
    

#training the nn
model = TinySLM(vocab_size)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = nn.CrossEntropyLoss()

epochs = 500

for epoch in range(epochs):

    logits = model(X)

    B, T, C = logits.shape

    loss = loss_fn(
        logits.view(B * T, C),
        Y.view(B * T)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:

        print(
            f"Epoch {epoch}, Loss: {loss.item()}"
        )

#testing the model
context = torch.tensor(
    [[char_to_idx["P"]]]
)

for _ in range(100):

    logits = model(context)

    logits = logits[:, -1, :]

    probs = torch.softmax(
        logits,
        dim=-1
    )

    next_token = torch.multinomial(
        probs,
        num_samples=1
    )

    context = torch.cat(
        [context, next_token],
        dim=1
    )

generated = "".join(
    idx_to_char[i.item()]
    for i in context[0]
)

print(generated)