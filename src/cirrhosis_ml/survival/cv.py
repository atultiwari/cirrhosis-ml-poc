"""Repeated nested cross-validation for survival models.

Outer loop: repeated stratified K-fold (stratified on the event indicator). Inner loop: K-fold
grid search on the outer training fold only (scored by Harrell's C, the sksurv default). Each
outer test fold yields fold-level metrics, and each patient receives one out-of-fold (OOF)
predicted risk at the horizon per repeat.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

from cirrhosis_ml.evaluation.survival_metrics import SurvivalData, point_metrics
from cirrhosis_ml.survival.estimators import ModelSpec, survival_at

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVConfig:
    horizon: float = 5.0
    auc_times: tuple[float, ...] = (2.0, 5.0, 10.0)
    tau: float = 10.0
    ibs_times: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)  # yearly grid (Mayo S0)
    n_splits: int = 5
    n_repeats: int = 10
    inner_splits: int = 3
    seed: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.horizon not in self.ibs_times:
            raise ValueError("horizon must be one of ibs_times (risk is read from that grid)")


@dataclass(frozen=True)
class CVResult:
    fold_metrics: pd.DataFrame  # one row per (model, repeat, fold)
    oof_risk: pd.DataFrame  # rows = patients, columns = (model, repeat); risk at horizon
    best_params: pd.DataFrame  # one row per tuned (model, repeat, fold)

    def mean_oof_risk(self) -> pd.DataFrame:
        """Per-patient OOF risk averaged over repeats (columns = models)."""
        return self.oof_risk.T.groupby(level="model", sort=False).mean().T


def fit_spec(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, cfg: CVConfig,
             seed: int) -> tuple[BaseEstimator, dict]:
    """Fit one spec, tuning by inner CV when it has a grid. Returns (model, best params)."""
    estimator = spec.make(seed)
    if not spec.param_grid:
        return estimator.fit(X, y), {}
    search = GridSearchCV(
        estimator, dict(spec.param_grid),
        cv=KFold(cfg.inner_splits, shuffle=True, random_state=seed), n_jobs=cfg.n_jobs,
        error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def _fold_metrics(model, X_tr, y_tr, X_te, y_te, cfg: CVConfig) -> tuple[dict, np.ndarray]:
    times = np.asarray(cfg.ibs_times)
    surv = survival_at(model, X_te, times)
    risk_h = 1.0 - surv[:, cfg.ibs_times.index(cfg.horizon)]
    test = SurvivalData.from_arrays(y_te["time"], y_te["event"], model.predict(X_te))
    train = SurvivalData.from_arrays(y_tr["time"], y_tr["event"], np.zeros(len(y_tr)))
    metrics: dict[str, float] = {}
    try:
        metrics.update(point_metrics(test, cfg.auc_times, cfg.tau, train=train))
    except ValueError as exc:  # e.g. horizon beyond this fold's follow-up
        logger.warning("fold discrimination metrics unavailable: %s", exc)
    try:
        metrics["ibs"] = float(integrated_brier_score(y_tr, y_te, surv, times))
    except ValueError as exc:
        logger.warning("fold IBS unavailable: %s", exc)
    return metrics, risk_h


def run_nested_cv(X: pd.DataFrame, time_: Sequence[float], event: Sequence[bool],
                  specs: Sequence[ModelSpec], cfg: CVConfig) -> CVResult:
    """Run repeated nested CV for every spec on identical outer folds."""
    X = X.reset_index(drop=True)
    y = Surv.from_arrays(event=np.asarray(event, dtype=bool), time=np.asarray(time_, dtype=float))
    outer = RepeatedStratifiedKFold(n_splits=cfg.n_splits, n_repeats=cfg.n_repeats,
                                    random_state=cfg.seed)
    oof = {(s.name, r): np.full(len(X), np.nan) for s in specs for r in range(cfg.n_repeats)}
    rows, params = [], []
    for split_no, (tr, te) in enumerate(outer.split(X, y["event"])):
        repeat, fold = divmod(split_no, cfg.n_splits)
        for spec in specs:
            t0 = time.perf_counter()
            fold_seed = cfg.seed + split_no
            model, best = fit_spec(spec, X.iloc[tr], y[tr], cfg, seed=fold_seed)
            metrics, risk_h = _fold_metrics(model, X.iloc[tr], y[tr], X.iloc[te], y[te], cfg)
            oof[(spec.name, repeat)][te] = risk_h
            rows.append({"model": spec.name, "repeat": repeat, "fold": fold,
                         "n_test": len(te), "events_test": int(y["event"][te].sum()),
                         "fit_seconds": time.perf_counter() - t0, **metrics})
            if best:
                params.append({"model": spec.name, "repeat": repeat, "fold": fold, **best})
        logger.info("outer split %d/%d done", split_no + 1, cfg.n_splits * cfg.n_repeats)
    oof_df = pd.DataFrame(oof)
    oof_df.columns = pd.MultiIndex.from_tuples(oof_df.columns, names=["model", "repeat"])
    if oof_df.isna().any().any():
        raise RuntimeError("some patients received no out-of-fold prediction")
    return CVResult(pd.DataFrame(rows), oof_df, pd.DataFrame(params))


def summarise_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of each metric across outer folds, per model (order preserved)."""
    metric_cols = [c for c in fold_metrics.columns
                   if c not in {"model", "repeat", "fold", "n_test", "events_test"}]
    grouped = fold_metrics.groupby("model", sort=False)[metric_cols]
    return grouped.mean().round(4).add_suffix("_mean").join(
        grouped.std().round(4).add_suffix("_sd"))
