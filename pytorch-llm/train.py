from config import Config
from model.gpt import GPTLanguageModel
from tokenizer.tokenizer import CharTokenizer
from training.dataset import create_dataloaders
from training.trainer import Trainer


def main():
    config = Config()

    # Load the training text.
    with open(config.data_path, "r", encoding="utf-8") as file:
        text = file.read()

    if not text:
        raise ValueError("The training text file is empty.")

    # Build the character tokenizer.
    tokenizer = CharTokenizer(text)
    config.vocab_size = tokenizer.vocab_size

    # Encode the text and create data loaders.
    encoded_text = tokenizer.encode(text)

    train_loader, val_loader = create_dataloaders(
        token_ids=encoded_text,
        block_size=config.block_size,
        batch_size=config.batch_size,
    )

    # Initialize the model and trainer.
    model = GPTLanguageModel(config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        tokenizer=tokenizer,
    )

    # Train the model.
    trainer.train()


if __name__ == "__main__":
    main()
