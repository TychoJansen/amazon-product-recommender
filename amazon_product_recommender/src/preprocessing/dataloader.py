r"""DataLoader: A configurable class for loading and preprocessing tabular data.

This class loads CSV or JSON data based on a config dictionary, performs basic
data cleaning, handles missing values, drops unused columns, preprocesses
string columns, capitalizes column names, and removes duplicates when configured.

Designed for flexible ETL workflows, with all options controlled via the config
dictionary or JSON file.

Config example:
{
    "data_path": "your_file.csv",
    "file_type": "csv",
    "use_cols": ["Id", "ProductId", ...],
    "patterns": {
        "remove_characters": "[\"']",
        "remove_parentheses": "\\(.*?\\)",
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


class DataLoader:
    """Load and preprocess tabular data using a configuration dictionary."""

    REQUIRED_KEYS = ["data_path", "file_type", "use_cols", "patterns", "fillna", "drop_duplicates"]
    REQUIRED_PATTERN_KEYS = ["remove_characters", "remove_parentheses", "remove_extra_whitespace"]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the DataLoader.

        Args:
            config: Configuration dictionary containing all required keys.
        """
        self.config: Dict[str, Any] = config
        self._validate_config()
        self.path: str = config["data_path"]
        self.data_type: str = config.get("file_type", "json")
        self.df: Optional[pd.DataFrame] = None

    def _validate_config(self) -> None:
        """Validate that the config contains all required keys."""
        for key in self.REQUIRED_KEYS:
            if key not in self.config:
                raise KeyError(f"Missing key in config: {key}")
        for pkey in self.REQUIRED_PATTERN_KEYS:
            if pkey not in self.config["patterns"]:
                raise KeyError(f"Missing pattern key in config['patterns']: {pkey}")

    def load(self) -> "DataLoader":
        """Load the source data into a pandas DataFrame."""
        if self.data_type == "json":
            self.df = pd.read_json(self.path, lines=True)
        elif self.data_type == "csv":
            self.df = pd.read_csv(self.path)
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
        return self

    def preprocess(self) -> pd.DataFrame:
        """Run the preprocessing pipeline and return the processed DataFrame."""
        self._fill_missing_values()
        self._drop_unused_columns()
        self._uppercase_col_names()
        self._preprocess_string_cols()
        self._drop_duplicates()
        if self.df is None:
            raise ValueError("Data has not been loaded.")
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

    def _uppercase_col_names(self) -> None:
        """Convert all DataFrame column names to uppercase."""
        if self.df is None:
            return
        self.df.columns = [col.upper() for col in self.df.columns]

    def _preprocess_string_cols(self) -> None:
        """Preprocess string columns by cleaning whitespace, quotes, and parentheses."""
        if self.df is None:
            return
        patterns = self.config["patterns"]
        remove_quotes_pattern = patterns["remove_characters"]
        remove_parentheses_pattern = patterns["remove_parentheses"]
        remove_extra_whitespace_pattern = patterns["remove_extra_whitespace"]

        def clean_text(x: Any) -> str:
            if pd.isna(x):
                return ""
            value = str(x).lower().strip()
            value = re.sub(remove_quotes_pattern, "", value)
            value = re.sub(remove_parentheses_pattern, "", value)
            value = re.sub(remove_extra_whitespace_pattern, " ", value)
            return value

        for col in self.df.select_dtypes(include="object").columns:
            self.df[col] = self.df[col].apply(clean_text)

    def _drop_duplicates(self) -> None:
        """Drop duplicate rows if the configuration enables duplicate removal."""
        if self.df is None:
            return
        if self.config.get("drop_duplicates", False):
            self.df = self.df.drop_duplicates()
