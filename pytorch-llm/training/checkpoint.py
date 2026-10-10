import os
import torch


def save_checkpoint(
    model,
    optimizer,
    config,
    path,
    step=0,
    loss=None,
    tokenizer=None,
):
    """
    Save the model, optimizer, configuration, and training progress.

    Args:
        model: GPTLanguageModel instance.
        optimizer: Optimizer used during training.
        config: Model configuration.
        path: Destination checkpoint path.
        step: Current training step.
        loss: Most recent loss value, if available.
        tokenizer: Optional tokenizer whose vocabulary should be saved.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "config": vars(config).copy(),
        "step": step,
        "loss": loss,
    }

    # Save tokenizer vocabulary when available
    if tokenizer is not None:
        checkpoint["stoi"] = getattr(tokenizer, "stoi", None)
        checkpoint["itos"] = getattr(tokenizer, "itos", None)

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
):
    """
    Load model weights and, optionally, optimizer state.

    Args:
        path: Checkpoint file path.
        model: GPTLanguageModel instance already initialized with
               a compatible configuration.
        optimizer: Optional optimizer to restore.
        device: Device on which to load the checkpoint.

    Returns:
        Dictionary containing the saved training metadata.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Checkpoints must come from a trusted source.
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    print(f"Checkpoint loaded from: {path}")

    return {
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss"),
        "config": checkpoint.get("config", {}),
        "stoi": checkpoint.get("stoi"),
        "itos": checkpoint.get("itos"),
    }


if __name__ == "__main__":
    from config import Config
    from model.gpt import GPTLanguageModel

    config = Config()
    config.vocab_size = 100

    model = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    checkpoint_path = "checkpoints/test_checkpoint.pt"

    # Save a test checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        path=checkpoint_path,
        step=100,
        loss=1.25,
    )

    # Create a fresh model and optimizer
    restored_model = GPTLanguageModel(config)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=config.learning_rate,
    )

    # Restore the checkpoint
    metadata = load_checkpoint(
        path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    print("Restored step:", metadata["step"])
    print("Restored loss:", metadata["loss"])
    print("Checkpoint test passed!")
