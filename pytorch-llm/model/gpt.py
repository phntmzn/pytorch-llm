class GPTLanguageModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # Initialize embeddings, Transformer blocks,
        # normalization, and the output projection.

    def forward(self, idx, targets=None):
        # Return logits and loss.
        ...

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # Autoregressively generate token IDs.
        ...
