import yaml
import os
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a nested Python dictionary.
    Supports environment variable expansion (e.g. ${HOME}) in paths.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw_text = f.read()

    # Expand environment variables like ${HOME}
    expanded_text = os.path.expandvars(raw_text)
    config = yaml.safe_load(expanded_text)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {path}")

    return config
