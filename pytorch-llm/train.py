import torch
import torch.nn.functional as F
from pathlib import Path

from config import Config
from model.gpt import GPTLanguageModel
from tokenizer.tokenizer import CharTokenizer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

config = Config()

torch.manual_seed(1337)

# Choose device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# --------------------------------------------------
# 2. Load training data
# --------------------------------------------------

data_path = Path(config.data_path)

if not data_path.exists():
    raise FileNotFoundError(
        f"Training data not found: {data_path}\n"
        "Create data/input.txt and add some training text."
    )

text = data_path.read_text(encoding="utf-8")

if len(text) < 100:
    raise ValueError(
        "Training data is too short. Add more text to data/input.txt."
    )

print(f"Training text length: {len(text):,} characters")


# --------------------------------------------------
# 3. Initialize tokenizer
# --------------------------------------------------

tokenizer = CharTokenizer(text)

encoded = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long,
)

vocab_size = tokenizer.vocab_size

print(f"Vocabulary size: {vocab_size}")


# --------------------------------------------------
# 4. Split training and validation data
# --------------------------------------------------

split_index = int(0.9 * len(encoded))

train_data = encoded[:split_index]
val_data = encoded[split_index:]

if len(train_data) <= config.block_size:
    raise ValueError(
        "Training data must be longer than the context window."
    )

if len(val_data) <= config.block_size:
    raise ValueError(
        "Validation data must be longer than the context window."
    )


# --------------------------------------------------
# 5. Create batches
# --------------------------------------------------

def get_batch(split: str):
    source = train_data if split == "train" else val_data

    indices = torch.randint(
        len(source) - config.block_size,
        (config.batch_size,),
    )

    x = torch.stack([
        source[i:i + config.block_size]
        for i in indices
    ])

    y = torch.stack([
        source[i + 1:i + config.block_size + 1]
        for i in indices
    ])

    return x.to(device), y.to(device)


# --------------------------------------------------
# 6. Initialize model
# --------------------------------------------------

config.vocab_size = vocab_size

model = GPTLanguageModel(config).to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"Model parameters: {parameter_count:,}")


# --------------------------------------------------
# 7. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
)


# --------------------------------------------------
# 8. Estimate loss
# --------------------------------------------------

@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for split in ("train", "val"):
        losses = torch.zeros(10)

        for k in range(10):
            x, y = get_batch(split)

            _, loss = model(x, y)

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# --------------------------------------------------
# 9. Training loop
# --------------------------------------------------

print("\nStarting training...\n")

model.train()

for step in range(config.max_iters):

    # Evaluate progress periodically
    if step % config.eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {step:5d} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Validation loss: {losses['val']:.4f}"
        )

    # Sample a batch
    x, y = get_batch("train")

    # Forward pass
    _, loss = model(x, y)

    # Backpropagation
    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    # Optional gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    # Update parameters
    optimizer.step()


# --------------------------------------------------
# 10. Save checkpoint
# --------------------------------------------------

checkpoint_path = Path(config.checkpoint_path)

checkpoint_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(config),
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "final_loss": loss.item(),
    },
    checkpoint_path,
)

print(f"\nModel saved to: {checkpoint_path}")


# --------------------------------------------------
# 11. Generate a sample
# --------------------------------------------------

model.eval()

start_token = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device,
)

with torch.no_grad():
    generated = model.generate(
        start_token,
        max_new_tokens=300,
    )

generated_text = tokenizer.decode(
    generated[0].tolist()
)

print("\nGenerated text:\n")
print(generated_text)
