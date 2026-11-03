"""Calibration of predicted risk at a fixed horizon for censored survival data.

- Grouped calibration: patients split into quantile groups of predicted risk; observed risk per
  group = 1 - Kaplan-Meier survival at the horizon (with its 95% CI).
- Calibration-in-the-large: observed/expected ratio, O/E = KM risk (all) / mean predicted risk.
- Calibration slope: Cox model of the outcome (follow-up administratively censored at the
  horizon) on the complementary log-log of predicted risk, log(-log(1 - p)); ideal slope = 1
  (< 1 means predictions too extreme, i.e. overfitting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

RISK_CLIP = 1e-6  # keeps log(-log(1 - p)) finite for p in {0, 1}


@dataclass(frozen=True)
class CalibrationSummary:
    oe_ratio: float
    slope: float
    slope_ci: tuple[float, float]
    table: pd.DataFrame


def km_risk(time: np.ndarray, event: np.ndarray, horizon: float) -> tuple[float, float, float]:
    """1 - KM survival at ``horizon`` with 95% CI (lower, upper).

    NaN when follow-up ends before ``horizon`` with a censored patient (KM undefined there);
    if it ends with an event the KM curve has reached its final value and is reported.
    """
    if not follow_up_reaches(time, event, horizon):
        return np.nan, np.nan, np.nan
    kmf = KaplanMeierFitter().fit(time, event)
    surv = float(kmf.survival_function_at_times(horizon).iloc[0])
    ci = kmf.confidence_interval_survival_function_
    row = ci[ci.index <= horizon].iloc[-1]
    return 1 - surv, 1 - float(row.iloc[1]), 1 - float(row.iloc[0])


def follow_up_reaches(time: np.ndarray, event: np.ndarray, horizon: float) -> bool:
    """True if KM survival at ``horizon`` is identified (follow-up reaches it, or the last
    observation before it is an event)."""
    t, e = np.asarray(time, float), np.asarray(event)
    return bool(t.max() >= horizon or e[np.argmax(t)] == 1)


def truncate_at(time: np.ndarray, event: np.ndarray, horizon: float):
    """Administrative censoring at ``horizon`` (returns new arrays)."""
    return np.minimum(time, horizon), np.where(time > horizon, 0, event)


def calibration_slope(time, event, risk, horizon: float) -> tuple[float, tuple[float, float]]:
    t, e = truncate_at(np.asarray(time, float), np.asarray(event, int), horizon)
    p = np.clip(np.asarray(risk, float), RISK_CLIP, 1 - RISK_CLIP)
    df = pd.DataFrame({"t": t, "e": e, "cll": np.log(-np.log(1 - p))})
    cph = CoxPHFitter().fit(df, duration_col="t", event_col="e")
    s = cph.summary.loc["cll"]
    return float(s["coef"]), (float(s["coef lower 95%"]), float(s["coef upper 95%"]))


def calibrate(time: npt.ArrayLike, event: npt.ArrayLike, risk: npt.ArrayLike, horizon: float,
              n_groups: int = 10) -> CalibrationSummary:
    """Grouped calibration table, O/E ratio and calibration slope at ``horizon``."""
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    p = np.asarray(risk, float)
    if not (len(t) == len(e) == len(p)):
        raise ValueError("time, event and risk must have equal length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("risk must be probabilities in [0, 1]")
    groups = pd.qcut(pd.Series(p).rank(method="first"), n_groups, labels=False)
    rows = []
    for g in range(n_groups):
        idx = (groups == g).to_numpy()
        obs, lo, hi = km_risk(t[idx], e[idx], horizon)
        rows.append({"group": g + 1, "n": int(idx.sum()), "mean_predicted": p[idx].mean(),
                     "observed_km": obs, "observed_lo": lo, "observed_hi": hi})
    observed_all = km_risk(t, e, horizon)[0]
    slope, slope_ci = calibration_slope(t, e, p, horizon)
    return CalibrationSummary(observed_all / p.mean(), slope, slope_ci, pd.DataFrame(rows))
