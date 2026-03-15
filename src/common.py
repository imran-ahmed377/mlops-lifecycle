from __future__ import annotations  # Enable postponed annotation evaluation for forward references.

from pathlib import Path  # Use pathlib for safe, cross-platform path handling.
from typing import Any, Dict  # Provide type hints for generic dictionary return values.

import yaml  # Parse YAML configuration files.

# Shared helpers for reading config and ensuring directories are created when needed.
def load_params(path: str = "params.yaml") -> Dict[str, Any]:  # Load parameters from a YAML file path.
    with open(path, "r", encoding="utf-8") as f:  # Open the YAML file in text mode with UTF-8 encoding.
        return yaml.safe_load(f)  # Safely parse YAML content into Python objects and return it.


def ensure_parent_dir(file_path: str) -> None:  # Ensure the parent directory of a file path exists.
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)  # Create missing parent folders without failing if they already exist.


def ensure_dir(dir_path: str) -> None:  # Ensure a directory path exists.
    Path(dir_path).mkdir(parents=True, exist_ok=True)  # Create missing directories and do nothing if already present.
