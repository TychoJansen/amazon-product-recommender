"""Utility functions for data processing and model training.

Provides helpers for random seed management, embedding loading, and batch collation.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_embeddings(load_path: str, device: str = "cpu") -> Tuple[torch.Tensor, List[int]]:
    """Load pre-computed embeddings from disk.

    Args:
        load_path: Path to saved embeddings checkpoint.
        device: Device to load embeddings to. Defaults to "cpu".

    Returns:
        Tuple of (embedding_matrix, product_ids) where embedding_matrix is a
        tensor of shape (num_products, embedding_dim) and product_ids is the
        list of product indices.
    """
    checkpoint = torch.load(load_path, map_location=device)

    embedding_matrix = checkpoint["embeddings"]
    product_ids = checkpoint["product_ids"]

    return embedding_matrix, product_ids


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples with variable-length user histories.

    Pads user_items to the max length in the batch.

    Args:
        batch: List of sample dictionaries from dataset.

    Returns:
        Collated batch dictionary with padded tensors.
    """
    max_len = max(x["user_items"].shape[0] for x in batch)

    user_items = []
    pos_items = []
    neg_items = []

    for x in batch:
        u = x["user_items"]
        pad = max_len - u.shape[0]

        if pad > 0:
            u = torch.cat([u, torch.zeros(pad, u.shape[1])], dim=0)

        user_items.append(u)
        pos_items.append(x["pos_item"])
        neg_items.append(x["neg_item"])

    return {
        "user_items": torch.stack(user_items),
        "pos_item": torch.stack(pos_items),
        "neg_item": torch.stack(neg_items),
    }
