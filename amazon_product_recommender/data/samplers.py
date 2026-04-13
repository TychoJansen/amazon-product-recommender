"""Negative sampler for contrastive learning in recommendation systems.

This module provides the NegativeSampler class for efficiently sampling negative
items (products a user has not interacted with) during training, with support
for both uniform and popularity-based sampling strategies.
"""

from typing import Dict, Optional, Set

import numpy as np


class NegativeSampler:
    """Sample negative items for users, avoiding seen interactions."""

    def __init__(
        self,
        num_items: int,
        user_product_dict: Dict[int, Set[int]],
        popularity: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the negative sampler.

        Args:
            num_items: Total number of items.
            user_product_dict: Mapping user -> set of seen items.
            popularity: Optional item popularity distribution.
        """
        self.num_items: int = num_items
        self.user_product_dict: Dict[int, Set[int]] = user_product_dict

        self.all_items: np.ndarray = np.arange(num_items, dtype=np.int32)

        if popularity is not None:
            popularity = popularity.astype(np.float64)
            popularity = popularity + 1e-12  # avoid zeros
            self.popularity: Optional[np.ndarray] = popularity / popularity.sum()
        else:
            self.popularity = None

    def _sample_raw(self, size: int) -> np.ndarray:
        """Sample raw items without filtering.

        Args:
            size: Number of samples.

        Returns:
            Array of sampled item indices.
        """
        if self.popularity is None:
            return np.random.randint(0, self.num_items, size=size)

        return np.random.choice(
            self.all_items,
            size=size,
            p=self.popularity,
        )

    def sample(
        self,
        user_id: int,
        num_negatives: int = 1,
    ) -> np.ndarray:
        """Sample negatives for a specific user.

        Ensures sampled items are not in the user's seen set.

        Args:
            user_id: User index.
            num_negatives: Number of negatives to sample.

        Returns:
            Array of negative item indices.
        """
        seen: Set[int] = self.user_product_dict.get(user_id, set())

        negatives = []

        while len(negatives) < num_negatives:
            candidates = self._sample_raw(size=num_negatives)

            for item in candidates:
                if item not in seen:
                    negatives.append(int(item))
                    if len(negatives) == num_negatives:
                        break

        return np.array(negatives, dtype=np.int64)
