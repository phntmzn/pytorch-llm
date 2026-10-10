import unittest

from tokenizer.tokenizer import CharTokenizer


class TestCharTokenizer(unittest.TestCase):

    def setUp(self):
        """Create a tokenizer using a small sample of text."""
        self.text = "hello world"
        self.tokenizer = CharTokenizer(self.text)

    def test_vocabulary_size(self):
        """Vocabulary size should equal the number of unique characters."""
        expected_size = len(set(self.text))

        self.assertEqual(
            self.tokenizer.vocab_size,
            expected_size,
        )

    def test_vocabulary_mappings(self):
        """Each character should map to a unique integer ID."""
        self.assertEqual(
            len(self.tokenizer.stoi),
            self.tokenizer.vocab_size,
        )

        self.assertEqual(
            len(self.tokenizer.itos),
            self.tokenizer.vocab_size,
        )

        for character, token_id in self.tokenizer.stoi.items():
            self.assertEqual(
                self.tokenizer.itos[token_id],
                character,
            )

    def test_encode(self):
        """Encoding should produce valid integer token IDs."""
        encoded = self.tokenizer.encode("hello")

        self.assertEqual(len(encoded), len("hello"))

        for token_id in encoded:
            self.assertIsInstance(token_id, int)
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, self.tokenizer.vocab_size)

    def test_decode(self):
        """Decoding should recover the original text."""
        encoded = self.tokenizer.encode("hello world")
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded, "hello world")

    def test_encode_decode_round_trip(self):
        """Encoding and decoding should preserve the original text."""
        samples = [
            "hello world",
            "hello",
            " ",
            "world",
            "hello world hello",
        ]

        for sample in samples:
            with self.subTest(sample=sample):
                encoded = self.tokenizer.encode(sample)
                decoded = self.tokenizer.decode(encoded)

                self.assertEqual(decoded, sample)

    def test_repeated_characters(self):
        """Repeated characters should reuse the same token ID."""
        encoded = self.tokenizer.encode("llll")

        self.assertEqual(len(set(encoded)), 1)

    def test_empty_string(self):
        """An empty string should encode and decode to empty values."""
        encoded = self.tokenizer.encode("")
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(encoded, [])
        self.assertEqual(decoded, "")

    def test_unknown_character(self):
        """Unknown characters should raise an informative error."""
        with self.assertRaises(KeyError):
            self.tokenizer.encode("hello!")

    def test_case_sensitivity(self):
        """Uppercase and lowercase characters should remain distinct."""
        with self.assertRaises(KeyError):
            self.tokenizer.encode("Hello")

    def test_special_characters(self):
        """Spaces and punctuation present in the vocabulary should work."""
        text = "hello, world!"

        tokenizer = CharTokenizer(text)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        self.assertEqual(decoded, text)


if __name__ == "__main__":
    unittest.main()
