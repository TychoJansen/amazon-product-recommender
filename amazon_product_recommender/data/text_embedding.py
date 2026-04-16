"""Text encoding and product embedding utilities for recommendation systems.

Provides utilities for encoding text into embeddings using transformers,
building product embedding matrices, and aggregating review texts by product.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


class TextEncoder:
    """Unified encoder supporting HuggingFace BERT and SentenceTransformers.

    Automatically selects encoding strategy based on model_name:
    - SentenceTransformers models for semantic similarity
    - BERT models for general language understanding
    """

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        """Initialize the text encoder.

        Args:
            model_name: Hugging Face model identifier.
            device: Device to run model on. Defaults to cuda if available, else cpu.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Select encoder based on model type
        if "sentence-transformers" in model_name:
            self.mode = "sentence_transformer"
            self.model = SentenceTransformer(model_name, device=self.device)
        else:
            self.mode = "bert"
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension for the selected model.

        Returns:
            Embedding dimension as integer.
        """
        if self.mode == "sentence_transformer":
            return self.model.get_sentence_embedding_dimension()
        else:
            return self.model.config.hidden_size

    @staticmethod
    def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to transformer output.

        Args:
            last_hidden_state: Output hidden states from transformer model.
            attention_mask: Attention mask indicating valid positions.

        Returns:
            Pooled embeddings of shape (batch_size, hidden_size).
        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """Encode list of texts into embeddings.

        Args:
            texts: List of text strings to encode.
            max_length: Maximum sequence length for BERT models. Defaults to 128.

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim).
        """
        if self.mode == "sentence_transformer":
            embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
            return embeddings.cpu()

        # BERT mode
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

            embeddings = self.mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

            embeddings = torch.nan_to_num(embeddings)
            embeddings = F.normalize(embeddings, dim=1)

        return embeddings.cpu()


class ProductEmbeddingBuilder:
    """Builds embeddings for products using a provided text encoder.

    Encodes all product texts into a matrix suitable for:
    - FAISS similarity search
    - Neural retrieval models
    - Cosine similarity calculations
    """

    def __init__(self, encoder: TextEncoder, batch_size: int = 32) -> None:
        """Initialize the embedding builder.

        Args:
            encoder: TextEncoder instance for encoding texts.
            batch_size: Number of texts to encode per batch. Defaults to 32.
        """
        self.encoder = encoder
        self.batch_size = batch_size

    def build(self, product_texts: Dict[int, str], save_path: Optional[str] = None) -> Tuple[torch.Tensor, List[int]]:
        """Encode all product texts into an embedding matrix.

        Args:
            product_texts: Dictionary mapping product_idx to aggregated review text.
            save_path: Optional path to save embeddings checkpoint.

        Returns:
            Tuple of (embedding_matrix, product_ids) where embedding_matrix has
            shape (num_products, embedding_dim).

        Raises:
            ValueError: If product_texts is empty or dimension mismatch occurs.
            TypeError: If encoder does not return torch.Tensor.
        """
        if not product_texts:
            raise ValueError("product_texts is empty")

        product_ids: List[int] = list(product_texts.keys())
        texts: List[str] = list(product_texts.values())

        all_embeddings = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding products", unit="batch"):
            batch_texts = texts[i : i + self.batch_size]
            embeddings = self.encoder.encode(batch_texts)

            # Validate encoder output
            if not isinstance(embeddings, torch.Tensor):
                raise TypeError("Encoder must return a torch.Tensor")

            if embeddings.ndim != 2:
                raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

            all_embeddings.append(embeddings)

        embedding_matrix = torch.cat(all_embeddings, dim=0)
        if embedding_matrix.shape[0] != len(product_ids):
            raise ValueError("Mismatch between embeddings and product IDs")

        # Optional save
        if save_path:
            torch.save({"embeddings": embedding_matrix, "product_ids": product_ids}, save_path)

        return embedding_matrix, product_ids


class ProductTextAggregator:
    """Aggregates multiple reviews into a single product-level text.

    Uses a balanced strategy sampling high-score, low-score, and recent reviews
    to prevent bias toward only positive or negative reviews.
    """

    def __init__(self, max_reviews: int = 20, max_chars: int = 2000) -> None:
        """Initialize the text aggregator.

        Args:
            max_reviews: Maximum number of reviews to aggregate per product. Defaults to 20.
            max_chars: Maximum character length of aggregated text. Defaults to 2000.
        """
        self.max_reviews = max_reviews
        self.max_chars = max_chars

    def aggregate(self, df: pd.DataFrame) -> Dict[int, str]:
        """Aggregate reviews by product.

        Samples from high-score (50%), low-score (25%), and recent (25%) reviews.

        Args:
            df: DataFrame with columns: product_idx, score, time, review.

        Returns:
            Dictionary mapping product_idx to aggregated review text.
        """
        product_texts: Dict[int, str] = {}

        grouped = df.groupby("product_idx")

        for product_idx, group in grouped:
            # Sample from different review categories
            pos = group[group["score"] >= 4]
            neg = group[group["score"] <= 2]
            recent = group.nlargest(self.max_reviews, "time")

            sampled = pd.concat(
                [pos.head(self.max_reviews // 2), neg.head(self.max_reviews // 4), recent.head(self.max_reviews // 4)]
            ).drop_duplicates()

            reviews = sampled["review"].dropna().tolist()

            if not reviews:
                text = "no reviews"
            else:
                text = " ".join(reviews)

            text = text[: self.max_chars]
            product_texts[product_idx] = text

        return product_texts
