from __future__ import annotations # for Python 3.7+ to allow postponed evaluation of type annotations

from pathlib import Path
from typing import Any, Dict # for type hinting, Dict is a generic type that represents a dictionary with specified key and value types

import yaml

# This module provides utility functions for loading parameters from a YAML file and ensuring that directories exist.
def load_params(path: str = "params.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

"""
This module provides utility functions for loading parameters 
from a YAML file and ensuring that directories exist. The load_params 
function reads a YAML file and returns its contents as a dictionary. 
The ensure_parent_dir function creates the parent directory of a given 
file path if it does not already exist, while the ensure_dir function creates 
a specified directory if it does not already exist. Both functions use the 
pathlib library to handle file paths and directories, allowing for easy and 
efficient directory management.
"""
def ensure_dir(dir_path: str) -> None:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
