"""Recommendation models for the Amazon product recommender.

This package contains neural network model architectures for learning user and
product embeddings for recommendation.
"""

from models.two_tower import TwoTowerModel

__all__ = ["TwoTowerModel"]
