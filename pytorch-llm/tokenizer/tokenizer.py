class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))

        self.stoi = {
            ch: i for i, ch in enumerate(chars)
        }

        self.itos = {
            i: ch for i, ch in enumerate(chars)
        }

        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(
            self.itos[int(i)] for i in ids
        )
