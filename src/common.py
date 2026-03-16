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


def resolve_mlflow_tracking_uri(params: Dict[str, Any], project_root: str | Path | None = None) -> str:  # Build a stable MLflow tracking URI from config.
    mlflow_cfg = params.get("mlflow", {})  # Read optional MLflow configuration block.
    configured_uri = str(mlflow_cfg.get("tracking_uri", "")).strip()  # Allow explicit URI override when provided.
    if configured_uri:
        return configured_uri

    tracking_dir = str(mlflow_cfg.get("tracking_dir", "mlruns")).strip() or "mlruns"  # Default to local mlruns folder.
    tracking_path = Path(tracking_dir)
    if not tracking_path.is_absolute():  # Resolve relative path against repository root for consistency.
        root = Path(project_root) if project_root is not None else Path.cwd()
        tracking_path = root / tracking_path

    tracking_path = tracking_path.resolve()
    tracking_path.mkdir(parents=True, exist_ok=True)  # Ensure local file-store folder exists before MLflow writes to it.
    return tracking_path.as_uri()  # Return file:// URI required by MLflow's tracking APIs.
