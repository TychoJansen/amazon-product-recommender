"""Negative sampler for contrastive learning in recommendation systems.

This module provides the NegativeSampler class for sampling negative items
(products a user has not interacted with) during training.
"""

from typing import Dict, Set

import numpy as np


class NegativeSampler:
    """Sample negative items for contrastive learning.

    Ensures sampled items are not in the user's positive interaction set to
    maintain meaningful contrastive pairs during training.
    """

    def __init__(self, num_products: int, user_product_dict: Dict[int, Set[int]], n_negatives: int = 3) -> None:
        """Initialize the negative sampler.

        Pre-computes unseen items for each user for efficient sampling.

        Args:
            num_products: Total number of unique product indices.
            user_product_dict: Dictionary mapping user indices to sets of
                product indices they have interacted with.
            n_negatives: Number of negative samples per user. Default is 3.
        """
        self.num_products: int = num_products
        self.user_product_dict: Dict[int, Set[int]] = user_product_dict
        self.n_negatives: int = n_negatives
        self.all_items: np.ndarray = np.arange(num_products)

        # Pre-compute unseen items for each user
        self.unseen_items: Dict[int, np.ndarray] = {
            u: np.array(list(set(range(num_products)) - s), dtype=np.int32) for u, s in user_product_dict.items()
        }

    def sample(self, user_id: int) -> int:
        """Sample a single negative item for the given user.

        Args:
            user_id: The user index to sample a negative item for.

        Returns:
            The product index of a negative item.
        """
        candidates = self.unseen_items.get(user_id)
        if candidates is None or len(candidates) == 0:
            return int(np.random.choice(self.all_items))
        return int(np.random.choice(candidates))

    def batch_sample(self, user_ids: np.ndarray) -> np.ndarray:
        """Sample negative items for a batch of users.

        Args:
            user_ids: Array of user indices.

        Returns:
            Array of shape (len(user_ids), n_negatives) with negative product indices.
        """
        negatives = []
        for u in user_ids:
            candidates = self.unseen_items.get(u)
            if candidates is None or len(candidates) == 0:
                # Sample with replacement if no unseen items
                negs = np.random.choice(self.all_items, size=self.n_negatives, replace=True)
            else:
                # Sample with replacement only if needed
                replace = len(candidates) < self.n_negatives
                negs = np.random.choice(candidates, size=self.n_negatives, replace=replace)
            negatives.append(negs)
        return np.array(negatives)
