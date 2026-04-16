"""Amazon reviews data processor with helpfulness and score calculation.

Extends BasePreProcessor with Amazon-specific preprocessing:
- Helpfulness ratio computation
- Combined score calculation
- Review text aggregation from summary and text columns
"""

from typing import Any, Dict

import pandas as pd
from data.preprocessing.base_preprocessor import BasePreProcessor


class AmazonPreProcessor(BasePreProcessor):
    """Processor for Amazon review data with helpfulness scoring.

    Inherits general preprocessing from BasePreProcessor and adds
    Amazon-specific features like helpfulness ratio and score calculation.
    """

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """Initialize the Amazon preprocessor.

        Args:
            df: Input DataFrame with review data.
            config: Configuration dictionary for preprocessing steps.
        """
        super().__init__(df=df, config=config)

    @BasePreProcessor.basic_preprocessing
    def preprocess(self) -> pd.DataFrame:
        """Preprocess Amazon review data according to configuration.

        Applies basic preprocessing pipeline, then computes helpfulness
        and score metrics, and removes intermediate columns.

        Returns:
            Preprocessed DataFrame with cleaned and computed features.
        """
        self._calculate_helpfulness_ratio()
        self._calculate_score()
        self.df.drop(
            columns=["id", "profilename", "helpfulnessnumerator", "helpfulnessdenominator", "helpfulness_ratio"],
            inplace=True,
        )
        return self.df

    def _calculate_helpfulness_ratio(self) -> None:
        """Calculate helpfulness ratio from numerator and denominator.

        Handles division by zero by replacing 0 denominators with 1.
        """
        self.df["helpfulnessdenominator"] = self.df["helpfulnessdenominator"].replace(0, 1)
        self.df["helpfulness_ratio"] = self.df["helpfulnessnumerator"] / self.df["helpfulnessdenominator"]

    def _calculate_score(self) -> None:
        """Calculate combined score from rating and helpfulness ratio.

        Normalizes rating to [0,1] range and multiplies by helpfulness ratio.
        """
        self.df["score"] = (self.df["score"] - 1) / 4 * self.df["helpfulness_ratio"]

    def make_review_text(
        self, summary_col: str = "summary", text_col: str = "text", summary_weight: int = 1, review_col: str = "review"
    ) -> pd.DataFrame:
        """Combine summary and text columns into a single review text.

        Args:
            summary_col: Name of summary column. Defaults to "summary".
            text_col: Name of text column. Defaults to "text".
            summary_weight: How many times to repeat the summary. Defaults to 1.
            review_col: Name of output review column. Defaults to "review".

        Returns:
            DataFrame with combined review column and originals removed.
        """
        if self.df is None:
            return self.df

        self.df[review_col] = (self.df[summary_col] + " ") * summary_weight + self.df[text_col]
        self.df.drop(columns=[summary_col, text_col], inplace=True)
        return self.df
