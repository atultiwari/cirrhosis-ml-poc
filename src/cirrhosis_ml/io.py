"""Data I/O helpers with a data-governance guard.

`assert_readable` refuses paths under `data/restricted/` (or `data/processed/`) unless the
environment variable `CIRRHOSIS_ML_ALLOW_RESTRICTED=1` is set. Atul sets it when he runs a
pipeline on credentialed data on his own machine; it is never set in tests or by Claude.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from cirrhosis_ml.config import PROCESSED_DATA_DIR, RESTRICTED_DATA_DIR

ALLOW_RESTRICTED_ENV = "CIRRHOSIS_ML_ALLOW_RESTRICTED"


class RestrictedDataError(PermissionError):
    """Raised when code tries to read restricted data without explicit opt-in."""


def _is_within(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def is_restricted(path: Path | str) -> bool:
    """True if `path` lies under `data/restricted/` or `data/processed/`."""
    p = Path(path)
    return _is_within(p, RESTRICTED_DATA_DIR) or _is_within(p, PROCESSED_DATA_DIR)


def assert_readable(path: Path | str) -> Path:
    """Return `path` as a Path if reading it is allowed, else raise RestrictedDataError."""
    p = Path(path)
    if is_restricted(p) and os.environ.get(ALLOW_RESTRICTED_ENV) != "1":
        raise RestrictedDataError(
            f"Refusing to read {p}: restricted/processed data. Set {ALLOW_RESTRICTED_ENV}=1 "
            "only when running locally on credentialed data, and emit aggregate outputs only."
        )
    return p


def read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    """Read a CSV after the governance check."""
    p = assert_readable(path)
    if not p.is_file():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p, **kwargs)
