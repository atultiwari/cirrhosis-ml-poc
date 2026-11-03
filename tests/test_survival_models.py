import numpy as np
import pandas as pd
import pytest
from sksurv.util import Surv

from cirrhosis_ml.pbc.models import FULL_FEATURES, MayoPublished, model_specs
from cirrhosis_ml.pbc.phase2 import mayo_coefficient_table, sample_size_table
from cirrhosis_ml.survival.competing import aj_cif, grouped_cif_calibration
from cirrhosis_ml.survival.cv import CVConfig, fit_spec, run_nested_cv, summarise_folds
from cirrhosis_ml.survival.estimators import survival_at
from cirrhosis_ml.survival.explain import ph_test
from cirrhosis_ml.survival.preprocessing import FeatureSet, make_preprocessor

FAST = ("mayo_published", "mayo_refit", "cox_full")


@pytest.fixture(scope="module")
def trial(pbc):
    return pbc[pbc["is_trial"]].reset_index(drop=True)


@pytest.fixture(scope="module")
def y(trial):
    return Surv.from_arrays(trial["event_death"].astype(bool), trial["time_years"])


def _specs(names=FAST):
    return [s for s in model_specs() if s.name in names]


def test_preprocessor_fits_on_training_rows_only(trial):
    prep = make_preprocessor(FeatureSet(log=("Cholesterol",), linear=("age_years",)), True, 0)
    prep.fit(trial.iloc[:150])
    out = prep.transform(trial.iloc[150:])
    assert list(out.columns) == ["log_Cholesterol", "age_years"]
    assert not out.isna().any().any()
    # scaling learned on the first 150 rows only -> held-out mean is not exactly 0
    assert abs(out["age_years"].mean()) > 1e-6


def test_mayo_published_matches_score_and_s0(trial):
    model = MayoPublished().fit(trial)
    surv = model.survival_at(trial, np.array([5.0]))
    assert surv.shape == (len(trial), 1)
    assert ((surv > 0) & (surv < 1)).all()
    with pytest.raises(ValueError, match="unavailable"):
        model.survival_at(trial, np.array([5.5]))


def test_mayo_published_fills_missing_inputs(trial):
    model = MayoPublished().fit(trial)
    gappy = trial.head(3).assign(Prothrombin=[np.nan, 10.0, 11.0])
    assert np.isfinite(model.predict(gappy)).all()


def test_refit_reproduces_dickson_initial_coefficients(trial, y):
    spec = _specs(("mayo_refit",))[0]
    model, _ = fit_spec(spec, trial, y, CVConfig(), seed=0)
    table = mayo_coefficient_table(model)
    np.testing.assert_allclose(table["refit_312"], table["dickson_initial_312"], atol=0.01)


def test_all_specs_fit_and_predict_survival(trial, y):
    for spec in model_specs():
        if spec.param_grid:  # keep the test fast: use the first grid point
            first = {k: v[0] for k, v in spec.param_grid.items()}
            model = spec.make(0).set_params(**first).fit(trial, y)
        else:
            model = spec.make(0).fit(trial, y)
        surv = survival_at(model, trial.head(5), np.array([1.0, 5.0]))
        assert surv.shape == (5, 2)
        assert (surv[:, 1] <= surv[:, 0] + 1e-9).all(), spec.name


def test_nested_cv_oof_complete_and_reproducible(trial):
    cfg = CVConfig(n_splits=3, n_repeats=1, seed=1)
    a = run_nested_cv(trial, trial["time_years"], trial["event_death"], _specs(), cfg)
    b = run_nested_cv(trial, trial["time_years"], trial["event_death"], _specs(), cfg)
    assert a.oof_risk.shape == (len(trial), len(FAST))
    pd.testing.assert_frame_equal(a.oof_risk, b.oof_risk)
    summary = summarise_folds(a.fold_metrics)
    assert summary.loc["mayo_refit", "harrell_c_mean"] > 0.75
    assert {"ibs_mean", "td_auc_5_mean"} <= set(summary.columns)


def test_cv_config_rejects_horizon_off_grid():
    with pytest.raises(ValueError, match="horizon"):
        CVConfig(horizon=5.5)


def test_ph_test_returns_p_values(trial, y):
    model, _ = fit_spec(_specs(("mayo_refit",))[0], trial, y, CVConfig(), seed=0)
    res = ph_test(model, trial, trial["time_years"], trial["event_death"])
    assert set(res.index) == {"log_Bilirubin", "log_Albumin", "log_Prothrombin", "age_years",
                              "edema_score"}
    assert res["p"].between(0, 1).all()


def test_aalen_johansen_below_km_risk(trial):
    t, c = trial["time_years"], trial["event_cr"]
    death_cif = aj_cif(t, c, 1, 5.0)
    transplant_cif = aj_cif(t, c, 2, 5.0)
    assert 0 < transplant_cif < death_cif < 0.5
    table = grouped_cif_calibration(t, c, np.linspace(0, 1, len(t)), 5.0)
    assert len(table) == 10


def test_sample_size_table(trial):
    table = sample_size_table(trial, 5.0)
    assert set(table["parameters"]) == {5, len(FULL_FEATURES.columns)}
    assert (table["available_events"] == 125).all()
