from dataclasses import dataclass


@dataclass
class Config:
    # Model
    block_size: int = 128
    vocab_size: int = 0
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    max_iters: int = 3000
    learning_rate: float = 3e-4
    eval_interval: int = 300

    # Paths
    data_path: str = "data/input.txt"
    checkpoint_path: str = "checkpoints/model.pt"
