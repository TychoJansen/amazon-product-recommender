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
    higher scores than negative items.
    """

    def forward(self, user_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vec: torch.Tensor) -> torch.Tensor:
        """Compute BPR loss for a batch of triplets.

        Args:
            user_vec: User embedding vectors of shape (batch_size, embedding_dim).
            pos_vec: Positive item embedding vectors of shape (batch_size, embedding_dim).
            neg_vec: Negative item embedding vectors of shape (batch_size, embedding_dim).

        Returns:
            Scalar BPR loss value.
        """
        pos_score = (user_vec * pos_vec).sum(dim=1)
        neg_score = (user_vec * neg_vec).sum(dim=1)

        loss = F.softplus(-(pos_score - neg_score))
        return loss.mean()
