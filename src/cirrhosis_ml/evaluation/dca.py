"""Decision-curve analysis for survival outcomes at a fixed horizon (Vickers et al. 2008).

Net benefit at threshold probability pt, with Kaplan-Meier handling censoring:
    NB = P(risk >= pt) * [ (1 - S_sel(t)) - S_sel(t) * pt / (1 - pt) ]
where S_sel(t) is KM survival at the horizon among patients with predicted risk >= pt.
"Treat all" uses everyone; "treat none" is 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from lifelines import KaplanMeierFitter

DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.05, 0.61, 0.01), 2))


def _km_survival(time: np.ndarray, event: np.ndarray, horizon: float) -> float:
    kmf = KaplanMeierFitter().fit(time, event)
    return float(kmf.survival_function_at_times(horizon).iloc[0])


def net_benefit(time: npt.ArrayLike, event: npt.ArrayLike, risk: npt.ArrayLike, horizon: float,
                thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> np.ndarray:
    t, e, p = (np.asarray(a, float) for a in (time, event, risk))
    out = []
    for pt in thresholds:
        if not 0 < pt < 1:
            raise ValueError("thresholds must lie in (0, 1)")
        sel = p >= pt
        if not sel.any():
            out.append(0.0)
            continue
        s = _km_survival(t[sel], e[sel], horizon)
        out.append(sel.mean() * ((1 - s) - s * pt / (1 - pt)))
    return np.asarray(out)


def decision_curves(time: npt.ArrayLike, event: npt.ArrayLike,
                    risks: Mapping[str, npt.ArrayLike], horizon: float,
                    thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Net benefit per model plus treat-all / treat-none, indexed by threshold."""
    n = len(np.asarray(time))
    curves = {name: net_benefit(time, event, r, horizon, thresholds) for name, r in risks.items()}
    curves["treat_all"] = net_benefit(time, event, np.ones(n), horizon, thresholds)
    curves["treat_none"] = np.zeros(len(thresholds))
    return pd.DataFrame(curves, index=pd.Index(thresholds, name="threshold"))
