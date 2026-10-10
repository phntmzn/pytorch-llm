import unittest

import torch

from config import Config
from model.attention import CausalSelfAttention


class TestCausalSelfAttention(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.config.n_embd = 128
        self.config.n_head = 4
        self.config.dropout = 0.0

        self.model = CausalSelfAttention(self.config)
        self.model.eval()

    def test_output_shape(self):
        """Output shape should match input shape."""
        x = torch.randn(2, 16, self.config.n_embd)

        output = self.model(x)

        self.assertEqual(output.shape, x.shape)

    def test_single_token(self):
        """Attention should handle a sequence of one token."""
        x = torch.randn(1, 1, self.config.n_embd)

        output = self.model(x)

        self.assertEqual(output.shape, (1, 1, self.config.n_embd))
        self.assertTrue(torch.isfinite(output).all())

    def test_different_batch_sizes(self):
        """Attention should support different batch and sequence sizes."""
        for batch_size, seq_len in [(1, 8), (4, 16), (2, 32)]:
            with self.subTest(batch_size=batch_size, seq_len=seq_len):
                x = torch.randn(
                    batch_size,
                    seq_len,
                    self.config.n_embd,
                )

                output = self.model(x)

                self.assertEqual(
                    output.shape,
                    (batch_size, seq_len, self.config.n_embd),
                )

    def test_causal_masking(self):
        """
        Changing future tokens must not affect earlier outputs.

        This verifies that attention does not look ahead.
        """
        torch.manual_seed(42)

        x1 = torch.randn(1, 8, self.config.n_embd)
        x2 = x1.clone()

        # Change tokens after position 3.
        x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])

        with torch.no_grad():
            output1 = self.model(x1)
            output2 = self.model(x2)

        # Positions 0 through 3 must remain unchanged.
        torch.testing.assert_close(
            output1[:, :4, :],
            output2[:, :4, :],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_gradient_flow(self):
        """Gradients should propagate through the attention module."""
        x = torch.randn(
            2,
            8,
            self.config.n_embd,
            requires_grad=True,
        )

        output = self.model(x)
        loss = output.square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_head_configuration(self):
        """Embedding dimension must be divisible by the number of heads."""
        invalid_config = Config()
        invalid_config.n_embd = 127
        invalid_config.n_head = 4

        with self.assertRaises(AssertionError):
            CausalSelfAttention(invalid_config)


if __name__ == "__main__":
    unittest.main()
