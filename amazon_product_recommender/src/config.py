"""Configuration helpers for the Amazon product recommender.

This module provides the Config class for loading JSON configuration files,
searching upward through parent directories when needed, and exposing config
values with dict-like access.
"""

import json
import os
from typing import Any, Dict, Optional


class Config:
    """Load and validate application configuration from JSON files."""

    def __init__(self, filename: str, config_dir: str = "configs", search_levels: int = 5) -> None:
        """Initialize the config loader.

        Args:
            filename: Name of the config file (e.g. "config.json").
            config_dir: Directory where the config file is stored.
            search_levels: How many levels up to search for configs.
        """
        self.path: str = self._find_config_path(filename, config_dir, search_levels)
        self._config: Optional[Dict[str, Any]] = None
        self.load()

    def _find_config_path(self, filename: str, config_dir: str, search_levels: int) -> str:
        """Find the config file path by searching parent directories."""
        start_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

        dir_to_search = start_dir
        for _ in range(search_levels):
            potential_path = os.path.join(dir_to_search, config_dir, filename)
            if os.path.exists(potential_path):
                return potential_path

            dir_to_search = os.path.dirname(dir_to_search)
        raise FileNotFoundError(f"Config file not found after searching {search_levels} levels up from {start_dir}")

    def load(self) -> None:
        """Load the JSON config file from disk."""
        with open(self.path, "r", encoding="utf-8") as f:
            self._config = json.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a configuration value or the provided default."""
        return self._config.get(key, default) if self._config is not None else default

    def __getitem__(self, key: str) -> Any:
        """Return a config value by key using dict-like access."""
        if self._config is None:
            raise KeyError("Configuration has not been loaded.")
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Return True if the key exists in the loaded config."""
        return key in self._config if self._config is not None else False

    @property
    def data(self) -> Dict[str, Any]:
        """Return the full loaded configuration dictionary."""
        if self._config is None:
            return {}
        return self._config
