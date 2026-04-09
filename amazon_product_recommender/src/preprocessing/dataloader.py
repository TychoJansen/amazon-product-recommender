"""DataLoader: A configurable class for loading and preprocessing tabular data.

This class loads CSV or JSON data based on a config dictionary.

Designed for flexible ETL workflows, with all options controlled via the config
dictionary or JSON file.

Config example:
{
    "data_path": "your_file.csv",
    "file_type": "csv"
}
"""
from typing import Any, Dict

import pandas as pd


class DataLoader:
    """Load and preprocess tabular data using a configuration dictionary."""

    REQUIRED_KEYS = ["data_path", "file_type"]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the DataLoader.

        Args:
            config: Configuration dictionary containing all required keys.
        """
        self.config: Dict[str, Any] = config
        self._validate_config()
        self.path: str = config["data_path"]
        self.data_type: str = config.get("file_type", "json")

    def _validate_config(self) -> None:
        """Validate that the config contains all required keys."""
        for key in self.REQUIRED_KEYS:
            if key not in self.config:
                raise KeyError(f"Missing key in config: {key}")

    def load(self) -> pd.DataFrame:
        """Load the source data into a pandas DataFrame."""
        if self.data_type == "json":
            return pd.read_json(self.path, lines=True)
        elif self.data_type == "csv":
            return pd.read_csv(self.path)
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
