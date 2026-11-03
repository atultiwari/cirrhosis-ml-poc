"""Model explanation and assumption checks for fitted survival pipelines.

- SHAP (model-agnostic permutation explainer) on the pipeline's risk score, so explanations are
  in the original (raw-unit) predictor space and imputation stays inside the model.
- Permutation importance: drop in Harrell's C when a predictor is shuffled.
- Proportional-hazards check: Schoenfeld-residual test (lifelines, rank-transformed time) on the
  preprocessed design matrix of a Cox pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import shap
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

SHAP_BACKGROUND = 50


def shap_values(model: BaseEstimator, X: pd.DataFrame, columns: Sequence[str],
                seed: int) -> shap.Explanation:
    """Permutation SHAP values of ``model.predict`` (risk score) for ``columns`` of ``X``."""
    cols = list(columns)
    data = X.loc[:, cols].reset_index(drop=True)

    def predict(values: np.ndarray) -> np.ndarray:
        return model.predict(pd.DataFrame(values, columns=cols))

    background = shap.sample(data, SHAP_BACKGROUND, random_state=seed)
    explainer = shap.PermutationExplainer(predict, shap.maskers.Independent(background),
                                          seed=seed)
    return explainer(data, silent=True)


def permutation_importance_table(model: BaseEstimator, X: pd.DataFrame, y: np.ndarray,
                                 columns: Sequence[str], seed: int,
                                 n_repeats: int = 20) -> pd.DataFrame:
    """Mean (SD) decrease in Harrell's C when each predictor is permuted."""
    res = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed)
    names = list(X.columns)
    table = pd.DataFrame({"importance_mean": res.importances_mean,
                          "importance_sd": res.importances_std}, index=names)
    return table.loc[list(columns)].sort_values("importance_mean", ascending=False)


def ph_test(pipeline: BaseEstimator, X: pd.DataFrame, time: np.ndarray,
            event: np.ndarray) -> pd.DataFrame:
    """Schoenfeld test per covariate of a fitted Cox pipeline's design matrix (step 'prep')."""
    design = pipeline.named_steps["prep"].transform(X).reset_index(drop=True)
    df = design.assign(_t=np.asarray(time, float), _e=np.asarray(event, int))
    cph = CoxPHFitter().fit(df, duration_col="_t", event_col="_e")
    res = proportional_hazard_test(cph, df, time_transform="rank")
    return res.summary.loc[:, ["test_statistic", "p"]].sort_values("p")
