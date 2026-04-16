"""Two-tower neural network architecture for recommendation systems.

Implements a two-tower model with separate user and item towers that learn
to encode user history and items into a shared embedding space for scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """Two-tower model with user and item towers for collaborative filtering.

    The user tower aggregates user's historical item embeddings into a user vector,
    while the item tower projects individual item embeddings. Both are normalized
    in the same embedding space.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 128) -> None:
        """Initialize the two-tower model.

        Args:
            embedding_dim: Dimension of input item embeddings.
            hidden_dim: Hidden dimension for MLP layers. Defaults to 128.
        """
        super().__init__()

        # USER tower processes aggregated item embeddings
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # ITEM tower projects individual items
        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_user(self, user_item_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode user from aggregated item embeddings.

        Averages item embeddings from user history, excluding padded zeros,
        then projects through MLP and normalizes.

        Args:
            user_item_embeddings: Tensor of shape (batch_size, num_items, embedding_dim).

        Returns:
            Normalized user vectors of shape (batch_size, embedding_dim).
        """
        # Avoid division by padded zeros
        mask = (user_item_embeddings.abs().sum(dim=2) > 0).float()

        summed = (user_item_embeddings * mask.unsqueeze(-1)).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1).unsqueeze(1)

        user_vec = summed / counts
        user_vec = self.user_mlp(user_vec)
        return F.normalize(user_vec, dim=1)

    def encode_item(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode item embeddings through the item tower.

        Projects item embeddings through MLP and normalizes.

        Args:
            item_embeddings: Tensor of shape (batch_size, embedding_dim) or (embedding_dim,).

        Returns:
            Normalized item vectors of same shape as input.
        """
        x = self.item_mlp(item_embeddings)
        return F.normalize(x, dim=1)

    def forward(self, user_item_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between users and items.

        Args:
            user_item_embeddings: User's historical item embeddings, shape (batch_size, num_items, embedding_dim).
            item_embeddings: Candidate item embeddings, shape (batch_size, embedding_dim).

        Returns:
            Similarity scores, shape (batch_size,).
        """
        user_vec = self.encode_user(user_item_embeddings)
        item_vec = self.encode_item(item_embeddings)
        return (user_vec * item_vec).sum(dim=1)
