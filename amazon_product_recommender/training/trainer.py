"""Trainer for two-tower recommendation models with BPR loss.

Implements the training loop for the two-tower model using Bayesian Personalized
Ranking (BPR) loss with support for gradient clipping and learning rate scheduling.
"""

import logging
from typing import Any

import torch
from tqdm import tqdm


class Trainer:
    """Trainer for two-tower model with BPR loss.

    Handles training iterations, gradient updates, and metric logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Any,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            loss_fn: Loss function to use.
            device: Device to use ("cuda" or "cpu"). Defaults to "cuda".
            lr: Learning rate for optimizer. Defaults to 1e-3.
            weight_decay: Weight decay for AdamW. Defaults to 1e-5.
            grad_clip: Gradient clipping threshold. Defaults to 1.0.
        """
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        """Run a single training epoch.

        Args:
            loader: DataLoader for training batches.

        Returns:
            Average loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Training", leave=False):

            user_items = batch["user_items"].to(self.device)
            pos_item = batch["pos_item"].to(self.device)
            neg_item = batch["neg_item"].to(self.device)

            user_vec = self.model.encode_user(user_items)
            pos_vec = self.model.encode_item(pos_item)
            neg_vec = self.model.encode_item(neg_item)

            # Compute loss
            loss = self.loss_fn(user_vec, pos_vec, neg_vec)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def train(self, loader: torch.utils.data.DataLoader, epochs: int = 5) -> None:
        """Run full training loop for specified number of epochs.

        Args:
            loader: DataLoader for training batches.
            epochs: Number of epochs to train. Defaults to 5.
        """
        for epoch in range(epochs):
            loss = self.train_epoch(loader)

            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
