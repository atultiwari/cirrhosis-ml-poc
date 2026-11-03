"""Load and prepare the Mayo Clinic PBC dataset (UCI ML Repository id 878, CC BY 4.0).

418 patients referred to Mayo Clinic 1974-1984; IDs 1-312 were randomised in the
D-penicillamine vs placebo trial, IDs 313-418 consented to basic measurements only
(most labs missing). UCI encodes missing values as the string ``"NaNN"``.

Derived columns added by :func:`prepare_pbc` (original UCI columns are kept, cleaned):

- ``age_years``    Age (days) / 365.25
- ``time_years``   N_Days / 365.25
- ``event_death``  1 = death (Status D); 0 = censored (C) or transplanted (CL) — primary analysis
- ``event_cr``     competing-risk code: 0 censored, 1 death, 2 liver transplant (CL)
- ``is_trial``     True for the 312 randomised patients
- ``sex_male``, ``ascites``, ``hepatomegaly``, ``spiders`` — 0/1 (NaN if missing)
- ``edema_score``  N = 0, S = 0.5 (oedema without diuretics or resolved by diuretics),
                   Y = 1 (oedema despite diuretics) — the coding used by the Mayo risk score
- ``drug_dpen``    1 = D-penicillamine, 0 = placebo, NaN = non-trial
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cirrhosis_ml.config import PBC_CACHE_DIR
from cirrhosis_ml.io import read_csv

logger = logging.getLogger(__name__)

UCI_ID = 878
CACHE_FILENAME = "pbc_uci878.csv"
DAYS_PER_YEAR = 365.25
N_TOTAL = 418
N_TRIAL = 312
MISSING_TOKEN = "NaNN"

EXPECTED_COLUMNS: tuple[str, ...] = (
    "ID", "N_Days", "Status", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders",
    "Edema", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT",
    "Tryglicerides", "Platelets", "Prothrombin", "Stage",
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT",
    "Tryglicerides", "Platelets", "Prothrombin", "Stage",
)
REQUIRED_COMPLETE: tuple[str, ...] = ("ID", "N_Days", "Status", "Age", "Sex")
STATUS_CODES ={"C": 0, "D": 1, "CL": 2}  # competing-risk coding
EDEMA_CODES = {"N": 0.0, "S": 0.5, "Y": 1.0}
YES_NO = {"N": 0.0, "Y": 1.0}
DRUG_CODES = {"Placebo": 0.0, "D-penicillamine": 1.0}


def fetch_pbc(cache_dir: Path = PBC_CACHE_DIR, refresh: bool = False) -> pd.DataFrame:
    """Return the raw UCI 878 table, downloading it once and caching it as CSV.

    The cache makes later runs offline and reproducible; pass ``refresh=True`` to re-download.
    """
    cache_path = Path(cache_dir) / CACHE_FILENAME
    if cache_path.is_file() and not refresh:
        logger.info("Using cached PBC data: %s", cache_path)
        return read_csv(cache_path, dtype=str, keep_default_na=False)

    from ucimlrepo import fetch_ucirepo  # imported lazily: network dependency

    try:
        dataset = fetch_ucirepo(id=UCI_ID)
    except Exception as exc:  # ucimlrepo raises generic errors on network failure
        raise ConnectionError(
            f"Could not download UCI dataset {UCI_ID}; check internet or place "
            f"{CACHE_FILENAME} in {cache_dir}"
        ) from exc
    raw = dataset.data.original
    _validate_columns(raw)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".csv.tmp")
    raw.to_csv(tmp_path, index=False)
    os.replace(tmp_path, cache_path)  # atomic: an interrupted write never leaves a partial cache
    logger.info("Cached PBC data to %s", cache_path)
    # Re-read from disk so fresh and cached runs yield identical dtypes.
    return read_csv(cache_path, dtype=str, keep_default_na=False)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"PBC data is missing expected columns: {missing}")


def _clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Replace missing tokens with NaN and coerce numeric columns (returns a new frame)."""
    subset = raw.loc[:, list(EXPECTED_COLUMNS)]
    cleaned = subset.where(~subset.isin([MISSING_TOKEN, ""]), np.nan)
    numeric = {c: pd.to_numeric(cleaned[c], errors="raise") for c in NUMERIC_COLUMNS}
    return cleaned.assign(ID=pd.to_numeric(cleaned["ID"], errors="raise").astype(int), **numeric)


def _map_strict(series: pd.Series, mapping: dict[str, float], name: str) -> pd.Series:
    unknown = set(series.dropna().unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Unexpected values in {name}: {sorted(unknown)}")
    return series.map(mapping).astype(float)


def prepare_pbc(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw UCI table and add analysis columns. Does not modify ``raw``."""
    _validate_columns(raw)
    df = _clean_raw(raw)
    incomplete = [c for c in REQUIRED_COMPLETE if df[c].isna().any()]
    if incomplete:
        raise ValueError(f"Mandatory columns contain missing values: {incomplete}")

    event_cr = _map_strict(df["Status"], STATUS_CODES, "Status").astype(int)
    drug_dpen = _map_strict(df["Drug"], DRUG_CODES, "Drug")
    is_trial = drug_dpen.notna()
    if not is_trial.equals(df["ID"] <= N_TRIAL):
        raise ValueError("Trial flag from Drug disagrees with ID <= 312")

    return df.assign(
        age_years=df["Age"] / DAYS_PER_YEAR,
        time_years=df["N_Days"] / DAYS_PER_YEAR,
        event_cr=event_cr,
        event_death=(event_cr == STATUS_CODES["D"]).astype(int),
        is_trial=is_trial,
        drug_dpen=drug_dpen,
        sex_male=_map_strict(df["Sex"], {"F": 0.0, "M": 1.0}, "Sex"),
        ascites=_map_strict(df["Ascites"], YES_NO, "Ascites"),
        hepatomegaly=_map_strict(df["Hepatomegaly"], YES_NO, "Hepatomegaly"),
        spiders=_map_strict(df["Spiders"], YES_NO, "Spiders"),
        edema_score=_map_strict(df["Edema"], EDEMA_CODES, "Edema"),
    )


def load_pbc(cache_dir: Path = PBC_CACHE_DIR, refresh: bool = False) -> pd.DataFrame:
    """Fetch (or read cached) and prepare the PBC dataset."""
    return prepare_pbc(fetch_pbc(cache_dir=cache_dir, refresh=refresh))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache and summarise UCI 878 (PBC).")
    parser.add_argument("--refresh", action="store_true", help="re-download from UCI")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_pbc(refresh=args.refresh)
    logger.info(
        "PBC: n=%d, trial=%d, deaths=%d, transplants=%d",
        len(df), int(df["is_trial"].sum()), int(df["event_death"].sum()),
        int((df["event_cr"] == STATUS_CODES["CL"]).sum()),
    )


if __name__ == "__main__":
    main()
