import math
import os

import torch
import torch.nn as nn

from training.checkpoint import save_checkpoint


class Trainer:
    """
    Trainer for a GPT-style autoregressive language model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        tokenizer=None,
        device=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer

        # Automatically choose the best available device.
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        self.global_step = 0
        self.best_val_loss = float("inf")

    @torch.no_grad()
    def evaluate(self, max_batches=20):
        """
        Calculate average training and validation loss.

        Uses a limited number of batches to keep evaluation affordable.
        """
        was_training = self.model.training
        self.model.eval()

        results = {}

        try:
            for name, loader in (
                ("train", self.train_loader),
                ("val", self.val_loader),
            ):
                losses = []

                for batch_index, (x, y) in enumerate(loader):
                    if batch_index >= max_batches:
                        break

                    x = x.to(self.device)
                    y = y.to(self.device)

                    _, loss = self.model(x, targets=y)
                    losses.append(loss.item())

                results[name] = (
                    sum(losses) / len(losses)
                    if losses
                    else float("nan")
                )
        finally:
            self.model.train(was_training)

        return results

    def train(self, max_iters=None):
        """
        Train the model for a specified number of optimizer steps.
        """
        if max_iters is None:
            max_iters = self.config.max_iters

        if len(self.train_loader) == 0:
            raise ValueError(
                "Training DataLoader has no batches. "
                "Reduce batch_size or provide more training text."
            )

        if len(self.val_loader) == 0:
            raise ValueError(
                "Validation DataLoader has no batches. "
                "Provide more validation text."
            )

        self.model.train()
        train_iterator = iter(self.train_loader)

        for step in range(max_iters):
            try:
                x, y = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                x, y = next(train_iterator)

            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            _, loss = self.model(x, targets=y)

            # Backpropagation
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Prevent unusually large gradients from destabilizing training.
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            # Update parameters
            self.optimizer.step()

            self.global_step += 1

            # Periodically evaluate and save checkpoints.
            if (
                self.global_step % self.config.eval_interval == 0
                or step == 0
                or step == max_iters - 1
            ):
                losses = self.evaluate()

                train_loss = losses["train"]
                val_loss = losses["val"]

                print(
                    f"Step {self.global_step:>6} | "
                    f"Batch loss: {loss.item():.4f} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Validation loss: {val_loss:.4f}"
                )

                if math.isfinite(val_loss) and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                    best_path = self._best_checkpoint_path()

                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        config=self.config,
                        path=best_path,
                        step=self.global_step,
                        loss=val_loss,
                        tokenizer=self.tokenizer,
                    )

        # Save the final training state as well.
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            config=self.config,
            path=self.config.checkpoint_path,
            step=self.global_step,
            loss=loss.item(),
            tokenizer=self.tokenizer,
        )

        print("Training complete.")
        print(f"Final checkpoint: {self.config.checkpoint_path}")

        return self.model

    def _best_checkpoint_path(self):
        """
        Create a path for the best-validation-loss checkpoint.
        """
        base, ext = os.path.splitext(self.config.checkpoint_path)
        return f"{base}_best{ext or '.pt'}"
