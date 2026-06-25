"""
config_loader.py  —  Configuration file loading and access module
Loads all settings from config.yaml, providing dot-path access interface.
"""

import os
import yaml
from typing import Any


class ConfigLoader:
    """
    Loads and holds all configuration from config.yaml.
    Use get(key, default) with dot-path notation, e.g.:
        cfg.get("display.vehicle_length_m", 4.5)
    """

    def __init__(self, config_path: str = "./config.yaml"):
        self._path = config_path
        self._data: dict = {}
        self.load()

    def load(self) -> None:
        """Reload configuration from disk."""
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"Config file not found: {self._path}")
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a config value by dot-path, e.g. "display.vehicle_length_m".
        Returns default if path does not exist.
        """
        parts = key.split(".")
        node = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def get_sensor_config(self, sensor_id: str) -> dict:
        """
        Get mount configuration for a single sensor, sensor_id format: "S01" ~ "S12".
        Returns dict with x_m, y_m, z_m, yaw_deg, pitch_deg, fov_deg, label.
        """
        return self.get(f"sensors.{sensor_id}", {})

    def get_all_sensor_ids(self) -> list[str]:
        """Return list of all sensor IDs defined in config (e.g. ["S01","S02",...])."""
        sensors = self._data.get("sensors", {})
        return list(sensors.keys())

    def __repr__(self) -> str:
        return f"ConfigLoader(path={self._path!r})"
