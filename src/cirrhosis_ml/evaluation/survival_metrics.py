"""Discrimination metrics for survival risk scores, with patient-level bootstrap CIs.

Risk scores follow the convention "higher = higher risk of the event". Times are in the same
unit as the horizons (years in Track A). Reused by Phase 2 models and by Track B.

- Harrell's C: ``sksurv.metrics.concordance_index_censored``
- Uno's C (IPCW, truncated at ``tau``): ``sksurv.metrics.concordance_index_ipcw``; the censoring
  distribution is estimated by Kaplan-Meier on ``train`` (default: the evaluation sample).
  IPCW assumes non-informative censoring — questionable when transplant is censored (PBC).
- Time-dependent cumulative/dynamic AUC at fixed horizons: ``sksurv.metrics.cumulative_dynamic_auc``
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv

from cirrhosis_ml.config import N_BOOTSTRAP, RANDOM_SEED

logger = logging.getLogger(__name__)

CI_LEVEL = 0.95
MIN_OK_FRACTION = 0.95  # minimum share of successful bootstrap replicates


@dataclass(frozen=True)
class SurvivalData:
    """Validated arrays for one evaluation sample."""

    time: np.ndarray
    event: np.ndarray  # bool
    risk: np.ndarray

    @classmethod
    def from_arrays(
        cls, time: npt.ArrayLike, event: npt.ArrayLike, risk: npt.ArrayLike
    ) -> SurvivalData:
        t = np.asarray(time, dtype=float)
        e_raw = np.asarray(event)
        r = np.asarray(risk, dtype=float)
        if not (t.shape == e_raw.shape == r.shape) or t.ndim != 1:
            raise ValueError("time, event and risk must be 1-D arrays of equal length")
        if not np.isin(e_raw, (0, 1)).all():
            raise ValueError("event must be boolean or 0/1 (recode competing risks first)")
        e = e_raw.astype(bool)
        if not (np.isfinite(t).all() and np.isfinite(r).all()):
            raise ValueError("time and risk must be finite (no missing or infinite values)")
        if (t <= 0).any():
            raise ValueError("survival times must be > 0")
        if not e.any():
            raise ValueError("no events in sample; discrimination is undefined")
        return cls(time=t, event=e, risk=r)

    def take(self, idx: np.ndarray) -> SurvivalData:
        return SurvivalData(time=self.time[idx], event=self.event[idx], risk=self.risk[idx])

    @property
    def structured(self) -> np.ndarray:
        return Surv.from_arrays(event=self.event, time=self.time)


def harrell_c(data: SurvivalData) -> float:
    return float(concordance_index_censored(data.event, data.time, data.risk)[0])


def uno_c(data: SurvivalData, tau: float, train: SurvivalData | None = None) -> float:
    """Uno's C. ``train`` supplies the censoring distribution (default: ``data`` itself —
    appropriate for apparent performance; pass the development data for held-out evaluation)."""
    y_train = (train or data).structured
    return float(concordance_index_ipcw(y_train, data.structured, data.risk, tau=tau)[0])


def time_dependent_auc(
    data: SurvivalData, horizons: Sequence[float], train: SurvivalData | None = None
) -> dict[float, float]:
    """Cumulative/dynamic AUC at each horizon; ``train`` as in :func:`uno_c`."""
    y_train = (train or data).structured
    auc, _ = cumulative_dynamic_auc(
        y_train, data.structured, data.risk, np.asarray(horizons, dtype=float)
    )
    return {float(h): float(a) for h, a in zip(horizons, auc, strict=True)}


def _check_horizons(data: SurvivalData, horizons: Sequence[float], tau: float) -> None:
    max_time = data.time.max()
    too_late = [h for h in (*horizons, tau) if not 0 < h < max_time]
    if too_late:
        raise ValueError(
            f"horizons/tau {too_late} must lie within follow-up (0, {max_time:.2f})"
        )


def point_metrics(
    data: SurvivalData,
    horizons: Sequence[float],
    tau: float,
    train: SurvivalData | None = None,
) -> dict[str, float]:
    """All discrimination metrics for one sample, keyed by metric name."""
    _check_horizons(data, horizons, tau)
    aucs = time_dependent_auc(data, horizons, train)
    return {
        "harrell_c": harrell_c(data),
        f"uno_c_tau{tau:g}": uno_c(data, tau, train),
        **{f"td_auc_{h:g}": v for h, v in aucs.items()},
    }


def bootstrap_metrics(
    data: SurvivalData,
    horizons: Sequence[float],
    tau: float,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    train: SurvivalData | None = None,
    min_ok_fraction: float = MIN_OK_FRACTION,
) -> pd.DataFrame:
    """Point estimates with percentile bootstrap 95% CIs (patients resampled with replacement).

    ``train`` (fixed, not resampled) supplies the IPCW censoring distribution for held-out
    evaluation; by default each resample uses itself (apparent performance). Replicates where a
    metric is undefined are dropped and counted in ``n_boot_ok``; if fewer than
    ``min_ok_fraction`` of replicates succeed, a ValueError is raised.
    """
    estimate = point_metrics(data, horizons, tau, train)
    rng = np.random.default_rng(seed)
    n = len(data.time)
    replicates: list[dict[str, float]] = []
    for _ in range(n_boot):
        sample = data.take(rng.integers(0, n, size=n))
        try:
            replicates.append(point_metrics(sample, horizons, tau, train))
        except ValueError as exc:  # metric undefined in this resample
            logger.debug("bootstrap replicate skipped: %s", exc)
    n_failed = n_boot - len(replicates)
    if n_failed:
        logger.warning("%d of %d bootstrap replicates failed and were dropped", n_failed, n_boot)
    if len(replicates) < min_ok_fraction * n_boot:
        raise ValueError(f"only {len(replicates)}/{n_boot} bootstrap replicates succeeded")
    boot = pd.DataFrame(replicates, columns=list(estimate))
    alpha = (1 - CI_LEVEL) / 2
    return pd.DataFrame(
        {
            "estimate": pd.Series(estimate),
            "ci_lower": boot.quantile(alpha),
            "ci_upper": boot.quantile(1 - alpha),
            "n_boot_ok": boot.notna().sum(),
        }
    ).rename_axis("metric")
