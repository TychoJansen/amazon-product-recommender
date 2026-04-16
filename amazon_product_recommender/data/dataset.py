"""Amazon dataset implementation with BPR sampling strategy.

Provides AmazonDataset class for training two-tower models with:
- Strong negative sampling (popularity-based + random)
- User history aggregation
- Efficient batching with padding
"""

from collections import Counter
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AmazonDataset(Dataset):
    """Dataset for BPR training with strong negative sampling.

    Features:
    - Popularity-based negatives (harder negatives)
    - Random negatives (diversity)
    - Safe filtering (no seen items)
    - Memory-efficient sampling
    """

    def __init__(
        self,
        config: Dict,
        df: pd.DataFrame,
        embedding_matrix: torch.Tensor,
        product_ids: List[int],
        user_product_dict: Dict[int, Set[int]],
    ) -> None:
        """Initialize the dataset.

        Args:
            config: Configuration dictionary with sampling parameters.
            df: DataFrame with user_idx and product_idx columns.
            embedding_matrix: Precomputed product embeddings.
            product_ids: List of product indices (aligned with embedding matrix rows).
            user_product_dict: Mapping of user_idx to set of interacted product_idx.
        """
        self.config = config
        self.user_ids = df["user_idx"].values
        self.product_ids = df["product_idx"].values

        self.embedding_matrix = embedding_matrix
        self.product_id_to_row = {pid: i for i, pid in enumerate(product_ids)}

        self.all_items = np.array(product_ids)
        self.user_product_dict = user_product_dict

        # Popularity-based sampling for hard negatives
        item_counts = Counter(self.product_ids)
        items, counts = zip(*item_counts.items())
        self.pop_items = np.array(items)
        self.pop_probs = np.array(counts, dtype=np.float64)
        self.pop_probs /= self.pop_probs.sum()

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.user_ids)

    def sample_negative(self, user: int, pos_item: int) -> int:
        """Sample a negative item not in user's history.

        Hybrid strategy: 70% popular items (hard negatives), 30% random (diversity).

        Args:
            user: User index.
            pos_item: Positive item index.

        Returns:
            Negative item index.
        """
        seen_items = self.user_product_dict[user]
        config = self.config.get("negative_sampling", {})

        for _ in range(config.get("attempts", 10)):
            if np.random.rand() < config.get("pop_negative_prob", 0.7):
                neg_item = np.random.choice(self.pop_items, p=self.pop_probs)
            else:
                neg_item = np.random.choice(self.all_items)

            if neg_item != pos_item and neg_item not in seen_items:
                return neg_item

        while True:
            neg_item = np.random.choice(self.all_items)
            if neg_item != pos_item and neg_item not in seen_items:
                return neg_item

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with user_items, pos_item, and neg_item embeddings.
        """
        user = int(self.user_ids[idx])
        pos_item = int(self.product_ids[idx])
        neg_item = self.sample_negative(user, pos_item)
        user_items = list(self.user_product_dict[user])

        # Limit history for speed
        max_items = self.config.get("max_items_per_user", 20)
        if len(user_items) > max_items:
            user_items = np.random.choice(user_items, max_items, replace=False)

        user_rows = [self.product_id_to_row[i] for i in user_items]
        user_embs = self.embedding_matrix[user_rows]

        pos_row = self.product_id_to_row[pos_item]
        neg_row = self.product_id_to_row[neg_item]

        return {
            "user_items": user_embs,
            "pos_item": self.embedding_matrix[pos_row],
            "neg_item": self.embedding_matrix[neg_row],
        }
