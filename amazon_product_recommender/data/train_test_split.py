"""Train-test split utilities for recommender systems.

Implements common strategies like leave-k-out and datetime-based splits.
Provides a unified interface for splitting user-item interactions into training and test sets.

Splits interactions per user into train and test sets.
"""

from typing import Dict, Literal, Set, Tuple

import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    strategy: Literal["leave_last_k", "datetime_split"] = "leave_last_k",
    k: int = 1,
    datetime_col: str = "time",
    threshold: float = 0.8,
) -> Tuple[pd.DataFrame, Dict[int, Set[int]]]:
    """Unified train-test split dispatcher.

    This function routes to the correct splitting strategy.

    Args:
        df: interaction dataframe
        strategy: split strategy name
        k: number of last items (used in leave_k_out)
        datetime_col: timestamp column
        threshold: float between 0 and 1 for datetime split cutoff quantile

    Returns:
        train_df, test_dict

    Raises:
        ValueError: if the strategy is unknown or if the input dataframe is empty
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    if strategy == "leave_k_out":
        return leave_k_out_split(df, k=k, datetime_col=datetime_col)

    elif strategy == "datetime_split":
        return datetime_split(df, datetime_col=datetime_col, threshold=threshold)

    else:
        raise ValueError(f"Unknown strategy: {strategy}. " f"Supported: leave_k_out, datetime_split")


def leave_k_out_split(
    df: pd.DataFrame,
    k: int,
    datetime_col: str,
) -> Tuple[pd.DataFrame, Dict[int, Set[int]]]:
    """Split interactions per user by holding out the last K items.

    This generalizes:
    - k = 1 → Leave-One-Out
    - k > 1 → K-last evaluation split

    Args:
        df: DataFrame with columns [user_idx, product_idx, datetime_col]
        k: number of last interactions to use as test set
        datetime_col: column used for ordering interactions

    Returns:
        train_df: remaining interactions
        test_dict: user -> set of held-out items
    """
    df = df.sort_values(["user_idx", datetime_col], kind="mergesort")

    # Rank interactions per user (0 = oldest) and calculate user sizes
    df["_rank"] = df.groupby("user_idx").cumcount()
    user_sizes = df.groupby("user_idx")["_rank"].transform("max") + 1

    # Last k interactions per user
    test_mask = df["_rank"] >= (user_sizes - k)

    test_df = df[test_mask]
    train_df = df[~test_mask]
    test_dict = test_df.groupby("user_idx")["product_idx"].apply(lambda x: set(map(int, x))).to_dict()

    return (
        train_df[["user_idx", "product_idx"]],
        test_dict,
    )


def datetime_split(df: pd.DataFrame, datetime_col: str, threshold: float) -> Tuple[pd.DataFrame, Dict[int, Set[int]]]:
    """Split interactions per user based on a datetime threshold.

    Interactions before the threshold go to the training set, and those after
    go to the test set.

    Args:
        df: DataFrame with columns [user_idx, product_idx, datetime_col]
        datetime_col: column used for splitting interactions
        threshold: float between 0 and 1 to determine the cutoff quantile

    Returns:
        train_df: interactions before the cutoff
        test_dict: user -> set of items interacted with after the cutoff
    """
    if not pd.api.types.is_numeric_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    cutoff = df[datetime_col].quantile(threshold)
    train_df = df[df[datetime_col] <= cutoff]
    test_df = df[df[datetime_col] > cutoff]
    test_dict = test_df.groupby("user_idx")["product_idx"].apply(lambda x: set(map(int, x))).to_dict()

    return (
        train_df[["user_idx", "product_idx"]],
        test_dict,
    )
