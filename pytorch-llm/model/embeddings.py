import torch
import torch.nn as nn


class TokenPositionalEmbedding(nn.Module):
    """
    Converts token IDs into embeddings and adds positional embeddings.

    Input:
        idx: (batch_size, seq_len)

    Output:
        x: (batch_size, seq_len, n_embd)
    """

    def __init__(self, config):
        super().__init__()

        self.block_size = config.block_size
        self.n_embd = config.n_embd

        # Learnable token embeddings.
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        # Learnable positional embeddings.
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx):
        """
        Args:
            idx: Integer token IDs with shape (B, T).

        Returns:
            Embeddings with shape (B, T, C).
        """

        batch_size, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"block size {self.block_size}."
            )

        # Convert token IDs into vectors.
        tok_emb = self.token_embedding(idx)

        # Generate positions: 0, 1, 2, ..., T-1.
        positions = torch.arange(
            seq_len,
            device=idx.device,
        )

        # Convert positions into vectors.
        pos_emb = self.position_embedding(positions)

        # Combine token identity and position information.
        x = tok_emb + pos_emb

        return self.dropout(x)


if __name__ == "__main__":
    from config import Config

    config = Config()
    config.vocab_size = 100

    embedding = TokenPositionalEmbedding(config)

    # Example batch: 2 sequences, 16 tokens each.
    idx = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    output = embedding(idx)

    print("Input shape: ", idx.shape)
    print("Output shape:", output.shape)

    print(
        "Parameters:",
        sum(p.numel() for p in embedding.parameters()),
    )
