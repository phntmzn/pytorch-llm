import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import TokenPositionalEmbedding
from model.transformer import TransformerBlock


class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.block_size = config.block_size

        # Token and positional embeddings
        self.embeddings = TokenPositionalEmbedding(config)

        # Stack of Transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )

        # Final normalization
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Project hidden states to vocabulary logits
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        # Initialize model weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize linear layers and embeddings."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: Input token IDs, shape (batch_size, sequence_length).
            targets: Optional target token IDs with the same shape.

        Returns:
            logits: Shape (batch_size, sequence_length, vocab_size).
            loss: Cross-entropy loss, or None if targets are omitted.
        """
        # Convert token IDs into embeddings
        x = self.embeddings(idx)

        # Apply Transformer blocks
        x = self.blocks(x)

        # Normalize hidden states
        x = self.ln_f(x)

        # Predict a score for every vocabulary token
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape

            logits_flat = logits.reshape(
                batch_size * seq_len,
                vocab_size,
            )
            targets_flat = targets.reshape(
                batch_size * seq_len,
            )

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generate tokens autoregressively.

        Args:
            idx: Starting token IDs, shape (batch_size, sequence_length).
            max_new_tokens: Number of tokens to generate.

        Returns:
            Input tokens followed by generated tokens.
        """
        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                # Keep only the context that fits within block_size
                idx_cond = idx[:, -self.block_size:]

                # Predict the next token
                logits, _ = self(idx_cond)

                # Select logits for the final position
                logits = logits[:, -1, :]

                # Convert scores into probabilities
                probabilities = F.softmax(logits, dim=-1)

                # Sample the next token
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

                # Append the sampled token to the sequence
                idx = torch.cat((idx, next_token), dim=1)

        finally:
            # Restore the previous training/evaluation mode
            self.train(was_training)

        return idx


if __name__ == "__main__":
    from config import Config

    config = Config()
    config.vocab_size = 100

    model = GPTLanguageModel(config)

    # Test a batch of token sequences
    idx = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    logits, loss = model(idx, targets=idx)

    print("Input shape: ", idx.shape)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    print(
        f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Test autoregressive generation
    prompt = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=20)

    print("Generated shape:", generated.shape)
:::

### 2. Test the model

From the root of your `pytorch-llm` directory, run:

```bash
python -m model.gpt
