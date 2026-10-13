
class CharacterTokenizer:
    def __init__(self, text):
        # 1. Find all unique characters
        self.vocab = sorted(set(text))

        # 2. Character -> integer ID
        self.stoi = {
            char: idx
            for idx, char in enumerate(self.vocab)
        }

        # 3. Integer ID -> character
        self.itos = {
            idx: char
            for char, idx in self.stoi.items()
        }

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        # Convert each character into its ID
        try:
            return [self.stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(
                f"Unknown character: {error.args[0]!r}"
            ) from error

    def decode(self, ids):
        # Convert each ID back into its character
        try:
            return "".join(self.itos[idx] for idx in ids)
        except KeyError as error:
            raise ValueError(
                f"Unknown token ID: {error.args[0]}"
            ) from error


# Test our tokenizer

text = "the fox is happy"

tokenizer = CharacterTokenizer(text)

encoded = tokenizer.encode("the fox is happy")
decoded = tokenizer.decode(encoded)

print("Vocabulary:", tokenizer.vocab)
print("Vocabulary size:", tokenizer.vocab_size)
print("Character IDs:", tokenizer.stoi)
print("Encoded:", encoded)
print("Decoded:", decoded)

assert decoded == "the fox is happy"
assert tokenizer.encode(text) == [
    tokenizer.stoi[char] for char in text
]

print("All tests passed!")
