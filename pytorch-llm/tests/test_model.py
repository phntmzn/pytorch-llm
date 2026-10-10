import unittest

import torch

from config import Config
from model.gpt import GPTLanguageModel


class TestGPTLanguageModel(unittest.TestCase):

    def setUp(self):
        """Create a small GPT model for testing."""
        self.config = Config()
        self.config.block_size = 16
        self.config.vocab_size = 32
        self.config.n_embd = 32
        self.config.n_head = 4
        self.config.n_layer = 2
        self.config.dropout = 0.0

        self.model = GPTLanguageModel(self.config)

    def test_model_initialization(self):
        """The model should initialize with trainable parameters."""
        self.assertIsInstance(self.model, GPTLanguageModel)

        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        self.assertGreater(parameter_count, 0)

    def test_forward_output_shape(self):
        """Forward pass should return correctly shaped logits."""
        batch_size = 2
        sequence_length = 8

        idx = torch.randint(
            0,
            self.config.vocab_size,
            (batch_size, sequence_length),
        )

        logits, loss = self.model(idx)

        self.assertEqual(
            logits.shape,
            (
                batch_size,
                sequence_length,
                self.config.vocab_size,
            ),
        )
        self.assertIsNone(loss)

    def test_loss_calculation(self):
        """The model should calculate a finite cross-entropy loss."""
        idx = torch.randint(
            0,
            self.config.vocab_size,
            (2, 8),
        )

        targets = torch.randint(
            0,
            self.config.vocab_size,
            (2, 8),
        )

        logits, loss = self.model(idx, targets=targets)

        self.assertEqual(
            logits.shape,
            (2, 8, self.config.vocab_size),
        )
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_single_token_input(self):
        """The model should handle a single-token input."""
        idx = torch.randint(
            0,
            self.config.vocab_size,
            (1, 1),
        )

        logits, loss = self.model(idx)

        self.assertEqual(
            logits.shape,
            (1, 1, self.config.vocab_size),
        )
        self.assertIsNone(loss)

    def test_generation_length(self):
        """Generation should append the requested number of tokens."""
        self.model.eval()

        prompt = torch.randint(
            0,
            self.config.vocab_size,
            (1, 4),
        )

        max_new_tokens = 5

        generated = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
        )

        self.assertEqual(
            generated.shape,
            (1, 4 + max_new_tokens),
        )

        # Generation must preserve the original prompt.
        torch.testing.assert_close(
            generated[:, :prompt.shape[1]],
            prompt,
        )

    def test_generation_token_range(self):
        """Generated token IDs must be within the vocabulary."""
        self.model.eval()

        prompt = torch.zeros((1, 1), dtype=torch.long)

        generated = self.model.generate(
            prompt,
            max_new_tokens=10,
        )

        self.assertTrue(torch.all(generated >= 0))
        self.assertTrue(
            torch.all(generated < self.config.vocab_size)
        )

    def test_generation_beyond_context_window(self):
        """Generation should work when the sequence exceeds block_size."""
        self.model.eval()

        prompt = torch.randint(
            0,
            self.config.vocab_size,
            (1, self.config.block_size),
        )

        generated = self.model.generate(
            prompt,
            max_new_tokens=3,
        )

        self.assertEqual(
            generated.shape,
            (1, self.config.block_size + 3),
        )

    def test_generation_restores_training_mode(self):
        """Generation should restore the model's original mode."""
        self.model.train()

        prompt = torch.zeros((1, 1), dtype=torch.long)
        self.model.generate(prompt, max_new_tokens=2)

        self.assertTrue(self.model.training)

    def test_gradient_flow(self):
        """Backpropagation should produce finite gradients."""
        idx = torch.randint(
            0,
            self.config.vocab_size,
            (2, 8),
        )

        targets = torch.randint(
            0,
            self.config.vocab_size,
            (2, 8),
        )

        _, loss = self.model(idx, targets=targets)

        loss.backward()

        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(all(grad is not None for grad in gradients))
        self.assertTrue(
            all(torch.isfinite(grad).all() for grad in gradients)
        )

    def test_context_length_limit(self):
        """The forward pass should reject sequences beyond block_size."""
        idx = torch.randint(
            0,
            self.config.vocab_size,
            (1, self.config.block_size + 1),
        )

        with self.assertRaises(ValueError):
            self.model(idx)


if __name__ == "__main__":
    unittest.main()
