"""ID mapper for converting raw user and product identifiers to numeric indices.

This module provides the IdMapper class for mapping categorical user and product
identifiers to zero-indexed numeric values suitable for embedding layers, with
support for unknown IDs encountered during inference.
"""

from typing import Dict, Set

import pandas as pd


class IdMapper:
    """Maps raw user and product IDs to contiguous integer indices.

    Supports an <UNK> index for unseen users/products at inference time.
    """

    def __init__(self) -> None:
        """Initialize the ID mapper with empty mappings."""
        self.user2idx: Dict[str, int] = {}
        self.product2idx: Dict[str, int] = {}

        self.idx2user: Dict[int, str] = {}
        self.idx2product: Dict[int, str] = {}

        self.unk_user_id: int = -1
        self.unk_product_id: int = -1

    def fit(self, df: pd.DataFrame) -> None:
        """Fit mappings from DataFrame.

        Args:
            df: DataFrame with 'userid' and 'productid' columns.
        """
        users = df["userid"].unique()
        products = df["productid"].unique()

        self.user2idx = {u: i for i, u in enumerate(users)}
        self.product2idx = {p: i for i, p in enumerate(products)}

        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2product = {i: p for p, i in self.product2idx.items()}

        self.unk_user_id = len(self.user2idx)
        self.unk_product_id = len(self.product2idx)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw IDs into integer indices.

        Unknown IDs are mapped to <UNK> indices.

        Args:
            df: Input DataFrame with 'userid' and 'productid' columns.

        Returns:
            DataFrame with added 'user_idx' and 'product_idx' columns.
        """
        df = df.copy()

        df["user_idx"] = df["userid"].map(self.user2idx).fillna(self.unk_user_id).astype(int)

        df["product_idx"] = df["productid"].map(self.product2idx).fillna(self.unk_product_id).astype(int)

        return df

    @staticmethod
    def build_user_product_dict(df: pd.DataFrame) -> Dict[int, Set[int]]:
        """Build user → interacted products mapping.

        Args:
            df: DataFrame with 'user_idx' and 'product_idx' columns.

        Returns:
            Dictionary mapping user_idx → set(product_idx).
        """
        return df.groupby("user_idx")["product_idx"].agg(set).to_dict()
