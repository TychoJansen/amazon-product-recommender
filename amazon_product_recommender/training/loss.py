"""Bayesian Personalized Ranking (BPR) loss for pairwise learning.

Implements the BPR loss function that encourages the model to rank positive
items higher than negative items for each user.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss for contrastive learning.

    Computes pairwise ranking loss that encourages positive items to have
    higher scores than negative items. Optionally applies L2 regularization
    to prevent overfitting.
    """

    def __init__(self, reg_weight: float = 0.0) -> None:
        """Initialize the BPR loss function.

        Args:
            reg_weight: L2 regularization weight for embedding vectors.
                Defaults to 0.0 (no regularization).
        """
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, user_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vec: torch.Tensor) -> torch.Tensor:
        """Compute BPR loss.

        Computes pairwise ranking loss using softplus activation.
        Loss = -log(sigmoid(score(user, pos) - score(user, neg)))

        Optionally adds L2 regularization on embedding norms.

        Args:
            user_vec: User embedding vectors, shape (batch_size, embedding_dim).
            pos_vec: Positive item embedding vectors, shape (batch_size, embedding_dim).
            neg_vec: Negative item embedding vectors, shape (batch_size, embedding_dim).

        Returns:
            Scalar loss value (mean over batch).
        """
        pos_score = (user_vec * pos_vec).sum(dim=1)
        neg_score = (user_vec * neg_vec).sum(dim=1)

        loss = F.softplus(-(pos_score - neg_score)).mean()

        # Optional L2 regularization
        if self.reg_weight > 0:
            reg = (
                user_vec.norm(2, dim=1).pow(2) + pos_vec.norm(2, dim=1).pow(2) + neg_vec.norm(2, dim=1).pow(2)
            ).mean()
            loss = loss + self.reg_weight * reg

        return loss
