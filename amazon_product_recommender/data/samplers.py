"""Negative sampler for contrastive learning in recommendation systems.

This module provides the NegativeSampler class for efficiently sampling negative
items (products a user has not interacted with) during training, with support
for both uniform and popularity-based sampling strategies.
"""

from typing import Dict, Optional, Set

import numpy as np


class NegativeSampler:
    """Sample negative items for contrastive learning.

    Supports both uniform and popularity-based negative sampling. Efficiently
    filters products the user has already interacted with.
    """

    def __init__(
        self,
        num_products: int,
        user_product_dict: Dict[int, Set[int]],
        popularity: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the negative sampler.

        Args:
            num_products: Total number of unique product indices.
            user_product_dict: Dictionary mapping user indices to sets of
                product indices they have interacted with.
            popularity: Optional item popularity distribution for weighted sampling.
                Default is None (uniform sampling).
        """
        self.num_products: int = num_products
        self.user_product_dict: Dict[int, Set[int]] = user_product_dict

        self.all_items: np.ndarray = np.arange(num_products, dtype=np.int32)

        self.popularity: Optional[np.ndarray] = None
        if popularity is not None:
            popularity = popularity.astype(np.float64)
            self.popularity = popularity / popularity.sum()

    def sample_uniform(self, user_id: int) -> int:
        """Sample a random negative item uniformly.

        Args:
            user_id: The user index to sample a negative item for.

        Returns:
            The product index of a negative item.
        """
        seen = self.user_product_dict.get(user_id, set())

        while True:
            neg = np.random.randint(0, self.num_products)
            if neg not in seen:
                return int(neg)

    def sample_popularity(self, user_id: int) -> int:
        """Sample a negative item weighted by item popularity.

        Uses the popularity distribution if provided, otherwise falls back
        to uniform sampling.

        Args:
            user_id: The user index to sample a negative item for.

        Returns:
            The product index of a negative item.
        """
        seen = self.user_product_dict.get(user_id, set())

        while True:
            if self.popularity is None:
                neg = np.random.randint(0, self.num_products)
            else:
                neg = np.random.choice(self.all_items, p=self.popularity)

            if neg not in seen:
                return int(neg)

    def batch_sample(self, user_ids: np.ndarray, mode: str = "uniform") -> np.ndarray:
        """Sample negative items for a batch of users.

        Args:
            user_ids: Array of user indices.
            mode: Sampling mode, either 'uniform' or 'popularity'.
                Default is 'uniform'.

        Returns:
            Array of shape (batch_size,) with negative product indices.
        """
        negatives = np.empty(len(user_ids), dtype=np.int32)

        for i, u in enumerate(user_ids):
            if mode == "popularity":
                negatives[i] = self.sample_popularity(u)
            else:
                negatives[i] = self.sample_uniform(u)

        return negatives
