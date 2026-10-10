import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    Dataset for autoregressive language modeling.

    Each sample contains an input sequence and the same sequence
    shifted one token to the right as its prediction target.
    """

    def __init__(self, token_ids, block_size):
        """
        Args:
            token_ids: A list or 1D tensor of token IDs.
            block_size: Number of input tokens per sample.
        """
        self.token_ids = torch.as_tensor(
            token_ids,
            dtype=torch.long,
        )

        self.block_size = block_size

        if self.token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional sequence.")

        if block_size < 1:
            raise ValueError("block_size must be at least 1.")

        if len(self.token_ids) <= block_size:
            raise ValueError(
                f"Need more than {block_size} tokens, "
                f"but received {len(self.token_ids)}."
            )

    def __len__(self):
        # Each sample uses block_size input tokens and one extra
        # token to provide the final prediction target.
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index):
        # Get block_size + 1 consecutive tokens.
        chunk = self.token_ids[
            index:index + self.block_size + 1
        ]

        # Inputs contain all but the final token.
        x = chunk[:-1]

        # Targets contain all but the first token.
        y = chunk[1:]

        return x, y


def create_dataloaders(
    token_ids,
    block_size,
    batch_size,
    train_split=0.9,
    num_workers=0,
):
    """
    Split token IDs into training and validation datasets.

    Returns:
        train_loader, val_loader
    """
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1.")

    token_ids = torch.as_tensor(token_ids, dtype=torch.long)

    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a one-dimensional sequence.")

    if len(token_ids) <= block_size + 1:
        raise ValueError(
            "Not enough tokens to create training and validation datasets."
        )

    # Split chronologically to avoid randomly mixing the text.
    split_index = int(len(token_ids) * train_split)

    # Keep each split separate so samples do not cross the boundary.
    train_tokens = token_ids[:split_index]
    val_tokens = token_ids[split_index:]

    train_dataset = TextDataset(train_tokens, block_size)
    val_dataset = TextDataset(val_tokens, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Generate example token IDs for testing.
    tokens = torch.arange(1000)

    block_size = 8
    batch_size = 4

    train_loader, val_loader = create_dataloaders(
        token_ids=tokens,
        block_size=block_size,
        batch_size=batch_size,
    )

    # Inspect one batch.
    x, y = next(iter(train_loader))

    print("Input batch shape: ", x.shape)
    print("Target batch shape:", y.shape)

    print("\nFirst input sequence:")
    print(x[0].tolist())

    print("\nFirst target sequence:")
    print(y[0].tolist())

    # Verify next-token alignment.
    assert torch.equal(x[:, 1:], y[:, :-1])

    print("\nTraining samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Dataset test passed!")
