class CharTokenizer:
    """
    Character-level tokenizer for a GPT-style language model.

    Each unique character in the training text receives an integer ID.
    """

    def __init__(self, text):
        """
        Build the vocabulary from training text.

        Args:
            text: String containing the training corpus.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        # Sort characters to ensure deterministic vocabulary ordering.
        characters = sorted(set(text))

        self.vocab_size = len(characters)

        # Character-to-ID mapping.
        self.stoi = {
            character: index
            for index, character in enumerate(characters)
        }

        # ID-to-character mapping.
        self.itos = {
            index: character
            for index, character in enumerate(characters)
        }

    def encode(self, text):
        """
        Convert a string into a list of token IDs.

        Raises:
            KeyError: If the text contains an unknown character.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        return [self.stoi[character] for character in text]

    def decode(self, token_ids):
        """
        Convert token IDs back into a string.
        """
        try:
            return "".join(
                self.itos[int(token_id)]
                for token_id in token_ids
            )
        except KeyError as error:
            raise ValueError(
                f"Unknown token ID: {error.args[0]}"
            ) from error

    def save_vocabulary(self):
        """
        Return the tokenizer vocabulary for checkpoint storage.
        """
        return {
            "stoi": self.stoi.copy(),
            "itos": self.itos.copy(),
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_vocabulary(cls, stoi, itos):
        """
        Reconstruct a tokenizer from a saved vocabulary.

        This avoids rebuilding the vocabulary from new text during
        generation, which could change the token IDs.
        """
        tokenizer = cls.__new__(cls)

        tokenizer.stoi = dict(stoi)
        tokenizer.itos = {
            int(index): character
            for index, character in itos.items()
        }
        tokenizer.vocab_size = len(tokenizer.stoi)

        if len(tokenizer.itos) != tokenizer.vocab_size:
            raise ValueError("Inconsistent tokenizer vocabulary.")

        for character, token_id in tokenizer.stoi.items():
            if tokenizer.itos.get(token_id) != character:
                raise ValueError("Inconsistent tokenizer mappings.")

        return tokenizer


if __name__ == "__main__":
    text = "hello world!"

    tokenizer = CharTokenizer(text)

    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)

    print("Original text:", text)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Character-to-ID mapping:", tokenizer.stoi)
    print("Encoded text:", encoded)
    print("Decoded text:", decoded)

    assert decoded == "hello"
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""

    print("Tokenizer test passed!")
