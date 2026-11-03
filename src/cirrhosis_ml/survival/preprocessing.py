"""Leakage-safe preprocessing for survival models.

Everything here is an sklearn transformer placed *inside* the model Pipeline, so imputation and
scaling are learned on the training fold only. Order: log-transform skewed labs (stateless) ->
multivariate imputation (IterativeImputer, i.e. single chained-equations imputation) ->
optional standardisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

IMPUTER_MAX_ITER = 10


@dataclass(frozen=True)
class FeatureSet:
    """Columns grouped by how they are transformed before imputation."""

    log: tuple[str, ...] = ()
    linear: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        return self.log + self.linear


def _log_names(_transformer, names: Sequence[str]) -> list[str]:
    return [f"log_{n}" for n in names]


def make_preprocessor(features: FeatureSet, scale: bool, seed: int) -> Pipeline:
    """Select ``features``, log-transform, impute (chained equations), optionally scale."""
    steps_select = []
    if features.log:
        steps_select.append(
            ("log", FunctionTransformer(np.log, feature_names_out=_log_names), list(features.log))
        )
    if features.linear:
        steps_select.append(("lin", "passthrough", list(features.linear)))
    select = ColumnTransformer(steps_select, remainder="drop", verbose_feature_names_out=False)
    steps = [
        ("select", select),
        ("impute", IterativeImputer(max_iter=IMPUTER_MAX_ITER, random_state=seed)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")
