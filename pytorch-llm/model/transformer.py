import torch
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    A GPT-style Transformer block.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.feedforward = FeedForward(config)

    def forward(self, x):
        # Multi-head causal self-attention with a residual connection
        x = x + self.attention(self.ln1(x))

        # Feed-forward network with a residual connection
        x = x + self.feedforward(self.ln2(x))

        return x


if __name__ == "__main__":
    from config import Config

    config = Config()
    block = TransformerBlock(config)

    # Batch of 2 sequences, each containing 16 tokens
    x = torch.randn(2, 16, config.n_embd)

    output = block(x)

    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)
    print(
        f"Parameters: {sum(p.numel() for p in block.parameters()):,}"
    )

    # Verify that the output shape matches the input shape
    assert output.shape == x.shape

    print("Transformer block test passed!")
