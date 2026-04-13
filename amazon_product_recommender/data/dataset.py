"""PyTorch dataset for user-item interaction triplets with negative sampling.

This module provides the InteractionDataset class for creating batches of
user-positive-negative triples used in contrastive learning.
"""

from typing import Any, Dict

import torch
from data.samplers import NegativeSampler
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """Dataset yielding (user, positive, negatives) samples."""

    def __init__(
        self,
        user_ids: Any,
        item_ids: Any,
        sampler: NegativeSampler,
        num_negatives: int = 1,
    ) -> None:
        """Initialize dataset.

        Args:
            user_ids: Array-like user indices.
            item_ids: Array-like positive item indices.
            sampler: NegativeSampler instance.
            num_negatives: Number of negatives per sample.
        """
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)

        self.sampler = sampler
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        """Return number of interactions."""
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a training sample.

        Args:
            idx: Index of sample.

        Returns:
            Dictionary containing:
                - user_id: (1,)
                - pos_item: (1,)
                - neg_items: (num_negatives,)
        """
        user = int(self.user_ids[idx])
        pos = int(self.item_ids[idx])

        negs = self.sampler.sample(user, self.num_negatives)

        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "pos_item": torch.tensor(pos, dtype=torch.long),
            "neg_items": torch.tensor(negs, dtype=torch.long),
        }
