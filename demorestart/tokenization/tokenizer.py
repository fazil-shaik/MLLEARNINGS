from pathlib import Path


from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers
)

DATA_DIR = Path("data")


tokenizer = Tokenizer(
    models.BPE(
        unk_token="<UNK>"
    )
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)


trainer = trainers.BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ],
)


files = [
    str(DATA_DIR / "TinyStories-train.txt")
]

tokenizer.train(files=files,trainer=trainer)

tokenizer.save("tokenizer.json")

print("Tokenizer created.")

print(
    tokenizer.encode(
        "Hello, my name is Fazil."
    ).ids
)