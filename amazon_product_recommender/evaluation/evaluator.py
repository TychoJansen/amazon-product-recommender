"""Evaluator for two-tower recommender systems.

Computes Recall@K, HitRate@K, and NDCG@K using full-item ranking
with efficient matrix multiplication.
"""

from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn as nn


class Evaluator:
    """GPU-based evaluator for ranking metrics in recommender systems."""

    def __init__(
        self,
        model: nn.Module,
        num_items: int,
        user_product_dict: Dict[int, Set[int]],
        item_embeddings: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Initialize evaluator.

        Args:
            model: trained two-tower model
            num_items: total number of items
            user_product_dict: train interactions (for filtering)
            item_embeddings: precomputed item embeddings (num_items, dim)
            device: torch device (cuda or cpu)
        """
        self.model = model.to(device)
        self.num_items = num_items
        self.user_product_dict = user_product_dict
        self.device = device

        # IMPORTANT: do NOT normalize here unless training used cosine similarity
        self.item_embeddings = item_embeddings.to(device)

    def _ndcg_at_k(self, recs: np.ndarray, gt: Set[int], k: int) -> float:
        """Compute NDCG@K."""
        dcg = 0.0

        for i, item in enumerate(recs[:k]):
            if item in gt:
                dcg += 1.0 / np.log2(i + 2)

        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), k)))

        return dcg / idcg if idcg > 0 else 0.0

    @torch.no_grad()
    def _score_batch(self, user_ids: List[int]) -> torch.Tensor:
        """Compute user-item scores for a batch.

        Returns:
            Tensor of shape (batch_size, num_items)
        """
        user_tensor = torch.tensor(user_ids, device=self.device)

        user_vecs = self.model.encode_users(user_tensor)
        item_vecs = self.item_embeddings

        user_vecs = torch.nn.functional.normalize(user_vecs, dim=1)
        item_vecs = torch.nn.functional.normalize(item_vecs, dim=1)

        # dot product (NO normalization unless training uses cosine loss)
        scores = torch.matmul(user_vecs, item_vecs.T)

        return scores

    @torch.no_grad()
    def evaluate(
        self,
        test_dict: Dict[int, Set[int]],
        k: int = 10,
        batch_size: int = 512,
    ) -> Dict[str, float]:
        """Evaluate model on ranking metrics.

        Args:
            test_dict: user -> ground truth items
            k: top-K cutoff
            batch_size: batch size for evaluation

        Returns:
            dict with recall, hit rate, ndcg
        """
        self.model.eval()

        users = list(test_dict.keys())

        recalls, hits, ndcgs = [], [], []

        for i in range(0, len(users), batch_size):
            batch_users = users[i : i + batch_size]

            scores = self._score_batch(batch_users)

            # mask seen items
            for row, user in enumerate(batch_users):
                seen = self.user_product_dict.get(user, set())
                if seen:
                    scores[row, list(seen)] = -1e9

            topk = torch.topk(scores, k, dim=1).indices.cpu().numpy()

            for row, user in enumerate(batch_users):
                gt = test_dict[user]
                recs = topk[row]

                hit_set = set(recs) & gt

                recalls.append(len(hit_set) / len(gt))
                hits.append(float(len(hit_set) > 0))
                ndcgs.append(self._ndcg_at_k(recs, gt, k))

        return {
            f"recall@{k}": float(np.mean(recalls)),
            f"hit_rate@{k}": float(np.mean(hits)),
            f"ndcg@{k}": float(np.mean(ndcgs)),
        }
