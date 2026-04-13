"""Two-tower embedding model for recommendation systems.

This module implements the two-tower architecture where user and item embeddings
are learned separately and compared using cosine similarity for ranking.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """Dual embedding tower model for user-item recommendation.

    Maintains separate embedding spaces for users and products, normalizing
    embeddings for cosine similarity-based ranking.
    """

    def __init__(self, num_users: int, num_products: int, embedding_dim: int = 64) -> None:
        """Initialize the two-tower model.

        Args:
            num_users: Total number of unique users.
            num_products: Total number of unique products.
            embedding_dim: Dimension of the embedding vectors. Default is 64.
        """
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)

        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Encode user IDs to embedding vectors.

        Args:
            user_ids: Tensor of user indices.

        Returns:
            User embedding vectors of shape (batch_size, embedding_dim).
        """
        x = self.user_embedding(user_ids)
        return self.user_mlp(x)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Encode product IDs to embedding vectors.

        Args:
            item_ids: Tensor of product indices.

        Returns:
            Product embedding vectors of shape (batch_size, embedding_dim).
        """
        x = self.product_embedding(item_ids)
        return self.item_mlp(x)

    def forward(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor, neg_item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute embeddings for user-positive-negative triplets.

        Args:
            user_ids: Tensor of user indices.
            pos_item_ids: Tensor of positive product indices.
            neg_item_ids: Tensor of negative product indices.

        Returns:
            Tuple of normalized (user_vec, pos_vec, neg_vec) embeddings.
        """
        user_vec = self.encode_users(user_ids)
        pos_vec = self.encode_items(pos_item_ids)
        neg_vec = self.encode_items(neg_item_ids)

        # Normalize embeddings for cosine similarity
        user_vec = F.normalize(user_vec, dim=1)
        pos_vec = F.normalize(pos_vec, dim=1)
        neg_vec = F.normalize(neg_vec, dim=1)

        return user_vec, pos_vec, neg_vec

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Predict relevance scores for user-item pairs.

        Args:
            user_ids: Tensor of user indices.
            item_ids: Tensor of product indices.

        Returns:
            Relevance score for each user-item pair.
        """
        user_vec = F.normalize(self.encode_users(user_ids), dim=1)
        item_vec = F.normalize(self.encode_items(item_ids), dim=1)

        return (user_vec * item_vec).sum(dim=1)
