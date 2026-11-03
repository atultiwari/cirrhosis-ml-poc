"""Project paths, global constants and YAML config loading.

`DATA_ROOT` can be overridden with the environment variable of the same name so that
Atul can point scripts at `data/restricted/...` and run them locally. Development and
tests always use the open data under `data/open/`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
OPEN_DATA_DIR: Path = PROJECT_ROOT / "data" / "open"
RESTRICTED_DATA_DIR: Path = PROJECT_ROOT / "data" / "restricted"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

PBC_CACHE_DIR: Path = OPEN_DATA_DIR / "pbc"
PBC_REPORTS_DIR: Path = REPORTS_DIR / "pbc"

RANDOM_SEED: int = 42
N_BOOTSTRAP: int = 1000
MIN_CELL_COUNT: int = 10  # suppress aggregate cells with n < 10 on restricted data


def data_root(default: Path = OPEN_DATA_DIR) -> Path:
    """Return the data root from the `DATA_ROOT` env var, or `default`."""
    value = os.environ.get("DATA_ROOT")
    return Path(value).expanduser().resolve() if value else default


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML file from `configs/` (e.g. ``load_yaml("icd_codes.yaml")``)."""
    path = (CONFIGS_DIR / name).resolve()
    if not path.is_relative_to(CONFIGS_DIR.resolve()):
        raise ValueError(f"Config name must stay inside {CONFIGS_DIR}: {name!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if not isinstance(content, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping, got {type(content).__name__}")
    return content
