"""Config loader for default_config.yaml."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str = "config/default_config.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        return yaml.safe_load(f)
