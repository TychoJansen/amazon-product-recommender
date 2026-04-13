"""Training loop for the two-tower recommendation model.

This module provides the Trainer class for orchestrating model training with
BPR loss and Adam optimization, including support for mixed precision training.
"""

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.loss import BPRLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class Trainer:
    """Train a recommendation model using BPR loss.

    Handles the training loop, optimizer updates, and loss computation for
    the two-tower embedding model with optional mixed precision training.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str = "cpu",
        use_amp: bool = True,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: PyTorch recommendation model.
            lr: Learning rate for optimizer. Default is 1e-3.
            device: Device to train on ('cpu' or 'cuda'). Default is 'cpu'.
            use_amp: Enable mixed precision training (GPU only). Default is True.
        """
        self.device: torch.device = torch.device(device)
        self.model: nn.Module = model.to(self.device)

        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )

        self.criterion: BPRLoss = BPRLoss()

        self.use_amp: bool = use_amp and self.device.type == "cuda"
        self.scaler: torch.amp.GradScaler = torch.amp.GradScaler() if self.use_amp else None

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Execute one training epoch.

        Args:
            dataloader: PyTorch DataLoader for the training data.
            epoch: Current epoch number for logging.

        Returns:
            Average loss over all batches in the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):

            user = batch["user_id"].to(self.device, non_blocking=True)
            pos = batch["pos_item"].to(self.device, non_blocking=True)
            neg = batch["neg_items"].to(self.device, non_blocking=True)
            b, k = neg.shape

            user = user.repeat_interleave(k)
            pos = pos.repeat_interleave(k)
            neg = neg.view(-1)

            self.optimizer.zero_grad()

            # -------------------------
            # Forward pass (AMP optional)
            # -------------------------
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    user_vec, pos_vec, neg_vec = self.model(user, pos, neg)
                    loss = self.criterion(user_vec, pos_vec, neg_vec)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                user_vec, pos_vec, neg_vec = self.model(user, pos, neg)
                loss = self.criterion(user_vec, pos_vec, neg_vec)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, dataloader: DataLoader, epochs: int = 2) -> None:
        """Execute full training for multiple epochs.

        Args:
            dataloader: PyTorch DataLoader for the training data.
            epochs: Number of training epochs. Default is 2.
        """
        progress = 1
        for epoch in range(epochs):
            loss = self.train_epoch(dataloader, progress)
            progress += 1
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
