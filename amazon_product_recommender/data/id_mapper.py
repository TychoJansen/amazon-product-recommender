"""ID mapper for converting raw user and product identifiers to numeric indices.

This module provides the IdMapper class for mapping categorical user and product
identifiers to zero-indexed numeric values suitable for embedding layers, with
support for unknown IDs encountered during inference.
"""

from typing import Dict, Set

import pandas as pd


class IdMapper:
    """Map user and product IDs to zero-indexed numeric values.

    Supports mapping of unknown IDs to a special <UNK> token for handling
    unseen users/products during inference.
    """

    def __init__(self) -> None:
        """Initialize the ID mapper with empty mappings."""
        self.user2idx: Dict[str, int] = {}
        self.product2idx: Dict[str, int] = {}
        self.unk_user_id: int = -1
        self.unk_product_id: int = -1

    def fit(self, df: pd.DataFrame) -> None:
        """Learn the ID mappings from the input DataFrame.

        Creates mappings for all unique user and product IDs. Initializes
        special <UNK> tokens for unseen IDs.

        Args:
            df: DataFrame with 'userid' and 'productid' columns.
        """
        users = df["userid"].unique()
        products = df["productid"].unique()

        self.user2idx = {u: i for i, u in enumerate(users)}
        self.product2idx = {p: i for i, p in enumerate(products)}

        # Add special <UNK> token for unseen IDs during inference
        # These tokens are NOT used during training, only during inference
        self.unk_user_id = len(self.user2idx)
        self.unk_product_id = len(self.product2idx)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned mappings to transform IDs to numeric indices.

        Maps unknown IDs to the <UNK> token index.

        Args:
            df: DataFrame with 'userid' and 'productid' columns to transform.

        Returns:
            DataFrame with new 'user_idx' and 'product_idx' columns.
        """
        df = df.copy()

        df["user_idx"] = df["userid"].map(self.user2idx).fillna(self.unk_user_id).astype(int)

        df["product_idx"] = df["productid"].map(self.product2idx).fillna(self.unk_product_id).astype(int)

        return df

    @staticmethod
    def build_user_product_dict(df: pd.DataFrame) -> Dict[int, Set[int]]:
        """Build a dictionary mapping user indices to sets of product indices.

        Used for negative sampling to avoid recommending already-interacted items.

        Args:
            df: DataFrame with 'user_idx' and 'product_idx' columns.

        Returns:
            Dictionary where keys are user indices and values are sets of
            product indices that the user has interacted with.
        """
        return df.groupby("user_idx")["product_idx"].agg(set).to_dict()
