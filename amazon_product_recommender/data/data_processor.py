r"""DataProcessor: A configurable class for preprocessing tabular data.

This class performs basic data cleaning, handles missing values, drops unused columns, preprocesses
string and datetime columns, lowers column names, and removes duplicates when configured.

Designed for flexible ETL workflows, with all options controlled via the config
dictionary or JSON file.

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

import re
from typing import Any, Dict, Optional

import pandas as pd


class DataProcessor:
    """Process tabular data using a configuration dictionary."""

    REQUIRED_KEYS = ["use_cols", "patterns", "fillna", "drop_duplicates"]

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """Initialize the DataProcessor.

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

    def preprocess(self) -> pd.DataFrame:
        """Run the preprocessing pipeline and return the processed DataFrame."""
        self._fill_missing_values()
        self._drop_unused_columns()
        self._preprocess_string_cols()
        self._convert_datetime_columns()
        self._sort_by_time()
        self._lowercase_col_names()
        self._drop_duplicates()
        return self.df

    def get_data(self) -> Optional[pd.DataFrame]:
        """Return the processed pandas DataFrame."""
        return self.df

    def _drop_unused_columns(self) -> None:
        """Drop columns that are not specified in the configuration's use_cols."""
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
        """Preprocess string columns by cleaning whitespace, quotes, and parentheses."""
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
            # numeric (unix timestamp)
            if pd.api.types.is_numeric_dtype(series):
                # auto-detect seconds vs milliseconds
                if series.max() > 1e12:
                    self.df[col] = pd.to_datetime(series, unit="ms", errors="coerce")
                else:
                    self.df[col] = pd.to_datetime(series, unit="s", errors="coerce")

            # string / object
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
        """Drop duplicate rows if the configuration enables duplicate removal."""
        if self.df is None:
            return
        if self.config.get("drop_duplicates", False):
            self.df = self.df.drop_duplicates()
