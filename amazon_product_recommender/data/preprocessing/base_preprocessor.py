r"""Base data processor for tabular data cleaning and preprocessing.

Provides configurable preprocessing pipeline for data cleaning, missing value
handling, column selection, string/datetime processing, and deduplication.

Config example:
{
    "use_cols": ["Id", "ProductId", ...],
    "patterns": {
        "remove_characters": "[\"']",
        "remove_extra_whitespace": "\\s+"
    },
    "fillna": {
        "Summary": ""
    },
    "drop_duplicates": true
}
"""

import functools
import re
from typing import Any, Callable, Dict, Optional

import pandas as pd


class BasePreProcessor:
    """Process tabular data using a configuration dictionary."""

    REQUIRED_KEYS = ["use_cols", "patterns", "fillna", "drop_duplicates"]

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """Initialize the data processor.

        Args:
            df: The pandas DataFrame to preprocess.
            config: Configuration dictionary containing all required keys.
        """
        self.config: Dict[str, Any] = config
        self._validate_config()
        self.df: pd.DataFrame = df

    def _validate_config(self) -> None:
        """Validate that the config contains all required keys."""
        for key in self.REQUIRED_KEYS:
            if key not in self.config:
                raise KeyError(f"Missing key in config: {key}")

    @staticmethod
    def basic_preprocessing(method: Callable) -> Callable:
        """Build a decorator to run basic preprocessing steps in order.

        Applies standard sequence: fill missing values, drop unused columns,
        clean strings, convert datetimes, sort by time, lowercase column names,
        and drop duplicates.

        Args:
            method: Method to wrap.

        Returns:
            Wrapped method that runs preprocessing before the original method.
        """

        @functools.wraps(method)
        def wrapper(self: "BasePreProcessor", *args: Any, **kwargs: Any) -> Any:
            self._fill_missing_values()
            self._drop_unused_columns()
            self._preprocess_string_cols()
            self._convert_datetime_columns()
            self._sort_by_time()
            self._lowercase_col_names()
            self._drop_duplicates()
            return method(self, *args, **kwargs)

        return wrapper

    def get_data(self) -> Optional[pd.DataFrame]:
        """Return the processed pandas DataFrame.

        Returns:
            Processed DataFrame or None if not yet processed.
        """
        return self.df

    def _drop_unused_columns(self) -> None:
        """Drop columns not specified in the configuration's use_cols."""
        if self.df is None:
            return
        use_cols = self.config.get("use_cols", [])
        drop_cols = [col for col in self.df.columns if col not in use_cols]
        self.df = self.df.drop(columns=drop_cols)

    def _fill_missing_values(self) -> None:
        """Fill missing values based on the configuration's fillna mapping."""
        if self.df is None:
            return
        col_dict = self.config.get("fillna", {})
        for col, value in col_dict.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(value)

    def _preprocess_string_cols(self) -> None:
        """Preprocess string columns by cleaning whitespace, quotes, and special characters."""
        patterns = self.config["patterns"]

        compiled_patterns = [
            (name, re.compile(pattern)) for name, pattern in patterns.items() if pattern  # Skip empty patterns
        ]

        def clean_text(x: Any) -> str:
            if pd.isna(x):
                return ""
            value = str(x).strip()
            for _, regex in compiled_patterns:
                value = regex.sub("", value)
            return value

        for col in self.df.select_dtypes(include="object").columns:
            self.df[col] = self.df[col].apply(clean_text)

    def _convert_datetime_columns(self) -> None:
        """Convert configured columns to pandas datetime format.

        Supports:
        - Unix timestamps (seconds or milliseconds)
        - ISO date strings
        - Mixed safe conversion

        Config:
            datetime_cols: list of column names to convert
        """
        cols = self.config.get("datetime_cols", [])

        for col in cols:
            if col not in self.df.columns:
                continue

            series = self.df[col]

            # Numeric (Unix timestamp)
            if pd.api.types.is_numeric_dtype(series):
                # Auto-detect seconds vs milliseconds
                if series.max() > 1e12:
                    self.df[col] = pd.to_datetime(series, unit="ms", errors="coerce")
                else:
                    self.df[col] = pd.to_datetime(series, unit="s", errors="coerce")
            # String / object
            else:
                self.df[col] = pd.to_datetime(series, errors="coerce")

    def _sort_by_time(self) -> None:
        """Sort the DataFrame by the configured datetime column."""
        sort_col = self.config.get("sort_col", None)
        datetime_cols = self.config.get("datetime_cols", None)

        if sort_col:
            self.df = self.df.sort_values(sort_col)
        elif datetime_cols and len(datetime_cols) == 1:
            self.df = self.df.sort_values(datetime_cols[0])

    def _lowercase_col_names(self) -> None:
        """Convert all DataFrame column names to lowercase."""
        if self.df is None:
            return
        self.df.columns = [col.lower() for col in self.df.columns]

    def _drop_duplicates(self) -> None:
        """Drop duplicate rows if the configuration enables removal."""
        if self.df is None:
            return
        if self.config.get("drop_duplicates", False):
            self.df = self.df.drop_duplicates()
