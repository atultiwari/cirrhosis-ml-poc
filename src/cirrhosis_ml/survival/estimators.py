"""Model specifications for survival prediction: Cox, penalised Cox, RSF, gradient boosting.

A :class:`ModelSpec` bundles a pipeline factory (preprocessing + estimator) with an optional
hyperparameter grid that is tuned by inner cross-validation on the training fold only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis

from cirrhosis_ml.survival.preprocessing import FeatureSet, make_preprocessor

COX_RIDGE = 1e-4  # tiny ridge for numerical stability; effectively unpenalised
COXNET_ALPHAS = tuple(float(a) for a in np.logspace(-3, -0.5, 8))
RSF_TREES = 300
GBSA_LEARNING_RATE = 0.05
GBSA_SUBSAMPLE = 0.8


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    make: Callable[[int], BaseEstimator]  # seed -> unfitted estimator / pipeline
    param_grid: Mapping[str, list[Any]] | None = field(default=None)


def cox_spec(name: str, label: str, features: FeatureSet) -> ModelSpec:
    return ModelSpec(name, label, lambda seed: Pipeline([
        ("prep", make_preprocessor(features, scale=True, seed=seed)),
        ("model", CoxPHSurvivalAnalysis(alpha=COX_RIDGE, ties="breslow")),
    ]))


def coxnet_spec(features: FeatureSet) -> ModelSpec:
    return ModelSpec(
        "coxnet", "Elastic-net Cox",
        lambda seed: Pipeline([
            ("prep", make_preprocessor(features, scale=True, seed=seed)),
            ("model", CoxnetSurvivalAnalysis(fit_baseline_model=True, max_iter=100_000)),
        ]),
        {"model__l1_ratio": [0.1, 0.5, 0.9], "model__alphas": [[a] for a in COXNET_ALPHAS]},
    )


def rsf_spec(features: FeatureSet) -> ModelSpec:
    return ModelSpec(
        "rsf", "Random survival forest",
        lambda seed: Pipeline([
            ("prep", make_preprocessor(features, scale=False, seed=seed)),
            ("model", RandomSurvivalForest(n_estimators=RSF_TREES, random_state=seed, n_jobs=1)),
        ]),
        {"model__min_samples_leaf": [5, 10, 20], "model__max_features": ["sqrt", 0.5]},
    )


def gbsa_spec(features: FeatureSet) -> ModelSpec:
    return ModelSpec(
        "gbsa", "Gradient-boosted Cox",
        lambda seed: Pipeline([
            ("prep", make_preprocessor(features, scale=False, seed=seed)),
            ("model", GradientBoostingSurvivalAnalysis(
                learning_rate=GBSA_LEARNING_RATE, subsample=GBSA_SUBSAMPLE, random_state=seed)),
        ]),
        {"model__n_estimators": [100, 300], "model__max_depth": [1, 2]},
    )


def survival_at(model: BaseEstimator, X, times: np.ndarray) -> np.ndarray:
    """Predicted survival probabilities, shape (n_samples, len(times))."""
    if hasattr(model, "survival_at"):
        return np.asarray(model.survival_at(X, times))
    fns = model.predict_survival_function(X)
    return np.vstack([fn(times) for fn in fns])
