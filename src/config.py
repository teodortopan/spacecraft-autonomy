"""Config loading. INFRASTRUCTURE."""

from __future__ import annotations

import os
from typing import Optional

import yaml

_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "vehicle.yaml")


def load_config(path: Optional[str] = None) -> dict:
    """Load the YAML config. Defaults to ``config/vehicle.yaml`` at the repo root."""
    with open(path or _DEFAULT, "r") as f:
        return yaml.safe_load(f)
