import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Prevents each token from attending to future tokens.
    """

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0, (
            "n_embd must be divisible by n_head"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Compute queries, keys, and values simultaneously.
        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=True,
        )

        # Project concatenated attention heads.
        self.proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=True,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor of shape (batch_size, seq_len, n_embd)
        """

        batch_size, seq_len, n_embd = x.shape

        # 1. Calculate query, key, and value projections.
        qkv = self.qkv(x)

        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2. Split the embedding dimension into attention heads.
        q = q.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        # Shapes:
        # q, k, v: (batch_size, n_head, seq_len, head_dim)

        # 3. Scaled dot-product attention with a causal mask.
        # PyTorch prevents attention to future positions.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(
                self.dropout if self.training else 0.0
            ),
            is_causal=True,
        )

        # 4. Merge attention heads.
        out = out.transpose(1, 2).contiguous()

        out = out.view(
            batch_size,
            seq_len,
            n_embd,
        )

        # 5. Output projection and residual dropout.
        out = self.proj(out)
        out = self.resid_dropout(out)

        return out


if __name__ == "__main__":
    from config import Config

    config = Config()

    attention = CausalSelfAttention(config)

    x = torch.randn(
        2,
        16,
        config.n_embd,
    )

    output = attention(x)

    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)
    print(
        "Parameters:  ",
        sum(p.numel() for p in attention.parameters()),
    )
