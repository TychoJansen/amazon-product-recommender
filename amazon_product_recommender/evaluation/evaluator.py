"""Evaluator for two-tower recommender systems.

Computes Recall@K, HitRate@K, and NDCG@K metrics using full-item ranking
with efficient matrix multiplication for batch evaluation.
"""

from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn as nn


class Evaluator:
    """Evaluator for two-tower models with history-based user representation.

    Computes ranking metrics on test sets:
    - Recall@K: fraction of relevant items in top-K
    - HitRate@K: whether any relevant item is in top-K
    - NDCG@K: normalized discounted cumulative gain
    """

    def __init__(
        self,
        model: nn.Module,
        item_embeddings: torch.Tensor,
        product_ids: List[int],
        user_product_dict: Dict[int, Set[int]],
        device: torch.device,
        max_history: int = 20,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model: Trained two-tower model.
            item_embeddings: Precomputed product embeddings.
            product_ids: List of product indices (aligned with embeddings).
            user_product_dict: Mapping of user_idx to set of interacted product_idx.
            device: Device to run evaluation on.
            max_history: Max user history items to consider. Defaults to 20.
        """
        self.model = model.to(device)
        self.device = device
        self.user_product_dict = user_product_dict
        self.max_history = max_history

        self.model.eval()

        # Product index to embedding matrix row mapping
        self.product_id_to_row = {pid: i for i, pid in enumerate(product_ids)}

        self.raw_item_embeddings = item_embeddings.to(device)

        # Precompute item tower vectors
        with torch.no_grad():
            self.item_vecs = self.model.encode_item(self.raw_item_embeddings)

    def _ndcg_at_k(self, recs: np.ndarray, gt: Set[int], k: int) -> float:
        """Compute NDCG@K for a single user.

        Args:
            recs: Recommended item indices, sorted by score.
            gt: Ground truth (relevant) item set.
            k: Cutoff rank.

        Returns:
            NDCG score between 0 and 1.
        """
        dcg = 0.0
        for i, item in enumerate(recs[:k]):
            if item in gt:
                dcg += 1.0 / np.log2(i + 2)

        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), k)))
        return dcg / idcg if idcg > 0 else 0.0

    def _encode_user_from_history(self, user: int) -> torch.Tensor:
        """Encode user from their interaction history.

        Args:
            user: User index.

        Returns:
            User embedding vector of shape (embedding_dim,).
        """
        items = list(self.user_product_dict.get(user, []))

        if not items:
            # Fallback for users with no history
            return torch.zeros(self.item_vecs.shape[1], device=self.device)

        # Limit history size
        if len(items) > self.max_history:
            items = np.random.choice(items, self.max_history, replace=False)

        rows = [self.product_id_to_row[i] for i in items]
        item_embs = self.raw_item_embeddings[rows]

        user_vec = self.model.encode_user(item_embs.unsqueeze(0))

        return user_vec.squeeze(0)

    @torch.no_grad()
    def _score_batch(self, user_ids: List[int]) -> torch.Tensor:
        """Score all items for a batch of users.

        Args:
            user_ids: List of user indices to score.

        Returns:
            Score matrix of shape (batch_size, num_items).
        """
        batch_item_embs = []
        lengths = []

        # Collect user embeddings with padding
        for user in user_ids:
            items = list(self.user_product_dict.get(user, []))

            if not items:
                emb = torch.zeros(1, self.item_vecs.shape[1])
                batch_item_embs.append(emb)
                lengths.append(1)
                continue

            if len(items) > self.max_history:
                items = np.random.choice(items, self.max_history, replace=False)

            rows = [self.product_id_to_row[i] for i in items]
            emb = self.raw_item_embeddings[rows]  # (n, D)

            batch_item_embs.append(emb)
            lengths.append(len(rows))

        # Pad to max length
        max_len = max(lengths)
        padded = []

        for emb in batch_item_embs:
            pad_size = max_len - emb.shape[0]

            if pad_size > 0:
                pad = torch.zeros(pad_size, emb.shape[1], device=self.device)
                emb = torch.cat([emb.to(self.device), pad], dim=0)
            else:
                emb = emb.to(self.device)

            padded.append(emb)

        user_items_tensor = torch.stack(padded)  # (B, L, D)
        user_vecs = self.model.encode_user(user_items_tensor)  # (B, D)
        scores = torch.matmul(user_vecs, self.item_vecs.T)  # (B, N)

        return scores

    @torch.no_grad()
    def evaluate(
        self,
        test_dict: Dict[int, Set[int]],
        k: int = 10,
        batch_size: int = 512,
    ) -> Dict[str, float]:
        """Evaluate model on test set.

        Args:
            test_dict: Mapping of user_idx to set of ground truth items.
            k: Cutoff rank for metrics. Defaults to 10.
            batch_size: Batch size for evaluation. Defaults to 512.

        Returns:
            Dictionary with metrics:
            - "recall@k": Recall@K (0 to 1)
            - "hit_rate@k": HitRate@K (0 to 1)
            - "ndcg@k": NDCG@K (0 to 1)
        """
        self.model.eval()

        users = list(test_dict.keys())

        recalls, hits, ndcgs = [], [], []
        for i in range(0, len(users), batch_size):
            batch_users = users[i : i + batch_size]

            scores = self._score_batch(batch_users)

            # Mask already-seen items
            for row, user in enumerate(batch_users):
                seen = self.user_product_dict.get(user, set())
                if seen:
                    rows = [self.product_id_to_row[i] for i in seen if i in self.product_id_to_row]
                    scores[row, rows] = -1e9

            # Get top-K recommendations
            topk = torch.topk(scores, k, dim=1).indices.cpu().numpy()

            idx_to_product = {v: k for k, v in self.product_id_to_row.items()}
            for row, user in enumerate(batch_users):
                gt = test_dict[user]

                recs = [idx_to_product[i] for i in topk[row]]

                hit_set = set(recs) & gt

                recalls.append(len(hit_set) / len(gt))
                hits.append(float(len(hit_set) > 0))
                ndcgs.append(self._ndcg_at_k(np.array(recs), gt, k))

        return {
            f"recall@{k}": float(np.mean(recalls)),
            f"hit_rate@{k}": float(np.mean(hits)),
            f"ndcg@{k}": float(np.mean(ndcgs)),
        }
