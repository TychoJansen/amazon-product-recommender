"""PyTorch dataset for user-item interaction triplets with negative sampling.

This module provides the InteractionDataset class for creating batches of
user-positive-negative triples used in contrastive learning.
"""

from typing import Any, Dict

import numpy as np
import torch
from data.samplers import NegativeSampler
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """Create triplets of user, positive, and negative items for training.

    Yields triplets: (user, positive_item, negative_items) where negative items
    are sampled from products the user has not interacted with.
    """

    def __init__(self, user_ids: Any, product_ids: Any, sampler: NegativeSampler) -> None:
        """Initialize the interaction dataset.

        Args:
            user_ids: Array of user indices.
            product_ids: Array of product indices (positive items).
            sampler: NegativeSampler instance for sampling negative items.
        """
        self.user_ids: torch.Tensor = torch.tensor(user_ids, dtype=torch.long)
        self.product_ids: torch.Tensor = torch.tensor(product_ids, dtype=torch.long)
        self.sampler: NegativeSampler = sampler

    def __len__(self) -> int:
        """Return the number of user-item interaction pairs."""
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a triplet of user, positive item, and negative items.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary with 'user_id', 'pos_item', and 'neg_item' tensors.
        """
        user = self.user_ids[idx]
        pos = self.product_ids[idx]
        neg = self.sampler.batch_sample(np.array([int(user)]))[0]
        return {
            "user_id": user,
            "pos_item": pos,
            "neg_item": torch.tensor(neg, dtype=torch.long),
        }
