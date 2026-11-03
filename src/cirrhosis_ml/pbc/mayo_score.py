"""Mayo natural-history risk score for primary biliary cirrhosis (fixed formula, no fitting).

Source: Dickson ER, Grambsch PM, Fleming TR, Fisher LD, Langworthy A. Prognosis in primary
biliary cirrhosis: model for decision making. *Hepatology* 1989;10(1):1-7.
Verified against the scanned original (2026-11-03):

- p6 formula (rounded "Final Mayo model" coefficients, Table 2, n = 418)::

      R = 0.871·ln(bilirubin mg/dL) − 2.53·ln(albumin g/dL) + 0.039·age (years)
          + 2.38·ln(prothrombin time s) + 0.859·edema

  Unrounded Table 2 values: 0.8707, −2.533, 0.0394, 2.380, 0.8592. Logs are natural.
- Edema coding (p4): 0 = no edema and no diuretic therapy; 0.5 = edema without diuretics, or
  edema resolved with diuretics; 1 = edema despite diuretic therapy. (UCI 878: N/S/Y.)
- Survival (p6, Table 4): S(t) = S0(t) ** exp(R − 5.07), with 5.07 the mean R of the 418 patients
  and S0(t) given for t = 1..7 years only.
- Worked examples (p6): R = 3.12 → S(5 y) = 0.96; R = 8.39 → S(1 y) = 0.43.
- Albumin was measured by serum protein electrophoresis in the original study.

Caveat: the final model was fitted on all 418 Mayo patients (312 trial + 106 cross-validation),
and the initial model (Table 2, n = 312) on the trial patients. Evaluating either on UCI 878 is
therefore *apparent* performance, not external validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import pandas as pd

ArrayLike = npt.ArrayLike

# Rounded coefficients as printed in the paper's formula (p6) — the conventional Mayo score.
MAYO_COEFFICIENTS = MappingProxyType(
    {"log_bilirubin": 0.871, "log_albumin": -2.53, "age": 0.039, "log_prothrombin": 2.38,
     "edema": 0.859}
)
# Initial model fitted on the 312 trial patients only (Table 2, first column).
MAYO_INITIAL_COEFFICIENTS = MappingProxyType(
    {"log_bilirubin": 0.8792, "log_albumin": -3.053, "age": 0.0333, "log_prothrombin": 3.016,
     "edema": 0.7847}
)
MAYO_MEAN_R = 5.07
MAYO_BASELINE_SURVIVAL = MappingProxyType(
    {1: 0.970, 2: 0.941, 3: 0.883, 4: 0.833, 5: 0.774, 6: 0.721, 7: 0.651}
)
VALID_EDEMA = (0.0, 0.5, 1.0)


def _as_float_array(values: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains missing values; impute or drop before scoring")
    return arr


def _require_positive(arr: np.ndarray, name: str) -> None:
    if (arr <= 0).any():
        raise ValueError(f"{name} must be > 0 (log-transformed)")


def mayo_risk_score(
    age_years: ArrayLike,
    bilirubin_mg_dl: ArrayLike,
    albumin_g_dl: ArrayLike,
    prothrombin_s: ArrayLike,
    edema: ArrayLike,
    coefficients: Mapping[str, float] = MAYO_COEFFICIENTS,
) -> np.ndarray:
    """Return the Mayo PBC risk score R (higher = worse prognosis).

    Inputs are scalars or equal-length arrays. Raises ValueError on missing values,
    non-positive lab values, or edema not in {0, 0.5, 1}.
    """
    age = _as_float_array(age_years, "age_years")
    bili = _as_float_array(bilirubin_mg_dl, "bilirubin_mg_dl")
    alb = _as_float_array(albumin_g_dl, "albumin_g_dl")
    pt = _as_float_array(prothrombin_s, "prothrombin_s")
    ede = _as_float_array(edema, "edema")
    for arr, name in ((bili, "bilirubin_mg_dl"), (alb, "albumin_g_dl"), (pt, "prothrombin_s")):
        _require_positive(arr, name)
    if not np.isin(ede, VALID_EDEMA).all():
        raise ValueError(f"edema must be one of {VALID_EDEMA}")

    c = coefficients
    return (
        c["log_bilirubin"] * np.log(bili)
        + c["log_albumin"] * np.log(alb)
        + c["age"] * age
        + c["log_prothrombin"] * np.log(pt)
        + c["edema"] * ede
    )


def mayo_survival_probability(risk_score: ArrayLike, years: int) -> np.ndarray:
    """Predicted probability of surviving ``years`` (1-7) from the published S0(t)."""
    if years not in MAYO_BASELINE_SURVIVAL:
        raise ValueError(f"S0(t) is published for years {sorted(MAYO_BASELINE_SURVIVAL)} only")
    r = _as_float_array(risk_score, "risk_score")
    return MAYO_BASELINE_SURVIVAL[years] ** np.exp(r - MAYO_MEAN_R)


def mayo_score_from_pbc(
    df: pd.DataFrame, coefficients: Mapping[str, float] = MAYO_COEFFICIENTS
) -> pd.Series:
    """Compute R for a prepared PBC frame (see :func:`cirrhosis_ml.pbc.load.prepare_pbc`)."""
    score = mayo_risk_score(
        df["age_years"], df["Bilirubin"], df["Albumin"], df["Prothrombin"], df["edema_score"],
        coefficients=coefficients,
    )
    return pd.Series(score, index=df.index, name="mayo_r")
