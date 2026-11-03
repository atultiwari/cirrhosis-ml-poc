"""Competing-risk helpers (Aalen-Johansen cumulative incidence).

Used for the transplant-as-competing-event sensitivity analysis: compares a model's predicted
risk of death with the Aalen-Johansen cumulative incidence of death (which does not treat
transplanted patients as if they could still die at the untransplanted rate, as KM does).
No mature Python Fine-Gray implementation exists; see DECISIONS.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from lifelines import AalenJohansenFitter


def _fit_aj(time: npt.ArrayLike, event_code: npt.ArrayLike, cause: int) -> AalenJohansenFitter:
    # lifelines jitters tied times (with a warning); the jitter is seeded for reproducibility
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AalenJohansenFitter(seed=0).fit(
            np.asarray(time, float), np.asarray(event_code, int), event_of_interest=cause
        )


def aj_cif(time: npt.ArrayLike, event_code: npt.ArrayLike, cause: int,
           horizon: float) -> float:
    """Aalen-Johansen cumulative incidence of ``cause`` at ``horizon`` (codes: 0 = censored)."""
    cif = _fit_aj(time, event_code, cause).cumulative_density_
    return float(cif[cif.index <= horizon].iloc[-1, 0]) if (cif.index <= horizon).any() else 0.0


def cif_curves(time: npt.ArrayLike, event_code: npt.ArrayLike,
               causes: dict[int, str]) -> pd.DataFrame:
    """Cumulative incidence curves (columns = cause labels) on a shared yearly-ish grid."""
    t = np.asarray(time, float)
    grid = np.linspace(0, np.floor(t.max()), 200)
    out = {}
    for code, label in causes.items():
        cif = _fit_aj(t, event_code, code).cumulative_density_.iloc[:, 0]
        out[label] = cif.reindex(cif.index.union(grid)).ffill().fillna(0.0).loc[grid].to_numpy()
    return pd.DataFrame(out, index=pd.Index(grid, name="years"))


def grouped_cif_calibration(time: npt.ArrayLike, event_code: npt.ArrayLike,
                            risk: npt.ArrayLike, horizon: float, cause: int = 1,
                            n_groups: int = 10) -> pd.DataFrame:
    """Mean predicted risk vs Aalen-Johansen observed incidence per risk group."""
    t, c, p = (np.asarray(a) for a in (time, event_code, risk))
    groups = pd.qcut(pd.Series(p).rank(method="first"), n_groups, labels=False).to_numpy()
    rows = [{"group": g + 1, "mean_predicted": float(p[groups == g].mean()),
             "observed_aj": aj_cif(t[groups == g], c[groups == g], cause, horizon)}
            for g in range(n_groups)]
    return pd.DataFrame(rows)
