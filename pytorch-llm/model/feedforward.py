import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward neural network.

    Architecture:
        Linear(n_embd, 4 * n_embd)
        GELU
        Linear(4 * n_embd, n_embd)
        Dropout

    Input:
        x: (batch_size, seq_len, n_embd)

    Output:
        x: (batch_size, seq_len, n_embd)
    """

    def __init__(self, config):
        super().__init__()

        n_embd = config.n_embd
        dropout = config.dropout

        self.net = nn.Sequential(
            # Expand the embedding dimension.
            nn.Linear(n_embd, 4 * n_embd),

            # Introduce nonlinearity.
            nn.GELU(),

            # Project back to the original dimension.
            nn.Linear(4 * n_embd, n_embd),

            # Regularization during training.
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    from config import Config

    config = Config()

    model = FeedForward(config)

    # Example input: batch=2, sequence=16, embedding=128.
    x = torch.randn(2, 16, config.n_embd)

    output = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Parameters: {parameters:,}")
