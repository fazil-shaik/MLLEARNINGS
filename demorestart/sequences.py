
from tokenizer import CharacterTokenizer


def create_training_data(text, context_length):
    tokenizer = CharacterTokenizer(text)

    # Convert the entire text into token IDs
    encoded = tokenizer.encode(text)

    if len(encoded) <= context_length:
        raise ValueError(
            "Text must contain more characters than context_length."
        )

    # Every input predicts the next character
    x = encoded[:-1]
    y = encoded[1:]

    # Print the full shifted sequences
    print("Vocabulary:", tokenizer.stoi)
    print("Encoded:", encoded)
    print("Input IDs:", x)
    print("Target IDs:", y)

    # Create context windows
    examples = []

    for i in range(len(encoded) - context_length):
        input_chunk = encoded[i : i + context_length]
        target_chunk = encoded[i + 1 : i + context_length + 1]

        examples.append((input_chunk, target_chunk))

    return tokenizer, examples


text = "hello"
context_length = 3

tokenizer, examples = create_training_data(
    text,
    context_length
)

print("\nTraining examples:")

for input_ids, target_ids in examples:
    print("Input:", tokenizer.decode(input_ids))
    print("Target:", tokenizer.decode(target_ids))
    print("Input IDs:", input_ids)
    print("Target IDs:", target_ids)
    print()

# Basic tests
assert tokenizer.decode([tokenizer.stoi["h"]]) == "h"
assert len(examples) == len(text) - context_length
assert examples[0][0] == tokenizer.encode("hel")
assert examples[0][1] == tokenizer.encode("ell")

print("All tests passed!")
