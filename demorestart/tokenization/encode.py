import numpy as np

from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file(
    "tokenizer.json"
)


def encode_file(
    input_file,
    output_file
):

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    print(
        f"Encoding {input_file}..."
    )

    encoded = tokenizer.encode(text)

    ids = np.array(
        encoded.ids,
        dtype=np.uint16
    )

    ids.tofile(output_file)

    print(
        f"Tokens: {len(ids):,}"
    )


encode_file(
    "data/TinyStories-train.txt",
    "data/train.bin"
)

encode_file(
    "data/TinyStories-val.txt",
    "data/val.bin"
)