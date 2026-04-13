"""Training utilities for the Amazon product recommender.

This package contains loss functions, optimizers, and training loops for
learning the recommendation model.
"""

from training.loss import BPRLoss
from training.trainer import Trainer

__all__ = ["BPRLoss", "Trainer"]
