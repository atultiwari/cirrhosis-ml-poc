"""PBC feature sets and the model line-up for Track A Phase 2.

Candidate predictors are the baseline variables in UCI 878 (16 parameters; histologic stage
entered as a linear 1-4 term). Treatment (D-penicillamine) is not a predictor: it had no effect in
the trial and is undefined for non-trial patients. Skewed labs are log-transformed (as in the
Mayo model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from cirrhosis_ml.pbc.mayo_score import (
    MAYO_BASELINE_SURVIVAL,
    mayo_score_from_pbc,
    mayo_survival_probability,
)
from cirrhosis_ml.survival.estimators import (
    ModelSpec,
    cox_spec,
    coxnet_spec,
    gbsa_spec,
    rsf_spec,
)
from cirrhosis_ml.survival.preprocessing import FeatureSet

FULL_FEATURES = FeatureSet(
    log=("Bilirubin", "Albumin", "Prothrombin", "Cholesterol", "Copper", "Alk_Phos", "SGOT",
         "Tryglicerides"),
    linear=("age_years", "edema_score", "Platelets", "Stage", "sex_male", "ascites",
            "hepatomegaly", "spiders"),
)
MAYO_FEATURES = FeatureSet(log=("Bilirubin", "Albumin", "Prothrombin"),
                           linear=("age_years", "edema_score"))
MAYO_INPUTS = ("age_years", "Bilirubin", "Albumin", "Prothrombin", "edema_score")


class MayoPublished(BaseEstimator):
    """The published Mayo score as a fixed (never refitted) estimator.

    ``fit`` only learns training-fold medians to fill the rare missing input (2 prothrombin times
    in the non-trial patients). Survival is available at the published S0 years (1-7) only.
    """

    def fit(self, X: pd.DataFrame, y=None) -> MayoPublished:
        self.medians_ = X.loc[:, list(MAYO_INPUTS)].median()
        return self

    def _filled(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, list(MAYO_INPUTS)].fillna(self.medians_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return mayo_score_from_pbc(self._filled(X)).to_numpy()

    def survival_at(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        r = self.predict(X)
        cols = []
        for t in np.asarray(times, dtype=float):
            if not float(t).is_integer() or int(t) not in MAYO_BASELINE_SURVIVAL:
                raise ValueError(f"published Mayo S0(t) unavailable at t = {t}")
            cols.append(mayo_survival_probability(r, int(t)))
        return np.column_stack(cols)


def model_specs() -> list[ModelSpec]:
    """Benchmarks first, then ML models; order is the reporting order."""
    return [
        ModelSpec("mayo_published", "Mayo score (published, fixed)", lambda seed: MayoPublished()),
        cox_spec("mayo_refit", "Cox, Mayo variables (refitted)", MAYO_FEATURES),
        cox_spec("cox_full", "Cox, all 16 predictors", FULL_FEATURES),
        coxnet_spec(FULL_FEATURES),
        rsf_spec(FULL_FEATURES),
        gbsa_spec(FULL_FEATURES),
    ]
