import os
import torch

from config import Config
from model.gpt import GPTLanguageModel


def load_model(checkpoint_path, device):
    """Load a trained GPT model and its tokenizer vocabulary."""

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train your model first by running: python train.py"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Restore the configuration used during training.
    config_values = checkpoint.get("config", {})
    config = Config()

    for key, value in config_values.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Restore the tokenizer vocabulary.
    stoi = checkpoint.get("stoi")
    itos = checkpoint.get("itos")

    if stoi is None or itos is None:
        raise ValueError(
            "The checkpoint does not contain a tokenizer vocabulary. "
            "Train the model with tokenizer saving enabled."
        )

    # Reconstruct the model.
    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, stoi, itos, config


def encode(text, stoi):
    """Convert text into character token IDs."""
    unknown = sorted(set(text) - set(stoi))

    if unknown:
        raise ValueError(
            f"Your prompt contains characters not in the vocabulary: "
            f"{unknown!r}\n"
            "Try a different prompt using characters from the training data."
        )

    return [stoi[character] for character in text]


def decode(token_ids, itos):
    """Convert character token IDs back into text."""
    return "".join(itos[int(token_id)] for token_id in token_ids)


def main():
    # Select the best available device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    config = Config()
    checkpoint_path = config.checkpoint_path

    model, stoi, itos, config = load_model(
        checkpoint_path,
        device,
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nGPT Text Generator")
    print("Type a prompt and press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        prompt = input("You: ")

        if prompt.strip().lower() == "quit":
            print("Goodbye!")
            break

        if not prompt:
            prompt = "The"

        try:
            token_ids = encode(prompt, stoi)

            # The model needs at least one token.
            if not token_ids:
                token_ids = [0]

            idx = torch.tensor(
                [token_ids],
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                output = model.generate(
                    idx,
                    max_new_tokens=200,
                )

            generated_ids = output[0].tolist()
            generated_text = decode(generated_ids, itos)

            print(f"\nGPT: {generated_text}\n")

        except ValueError as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
