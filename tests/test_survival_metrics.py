import numpy as np
import pytest

from cirrhosis_ml.evaluation.survival_metrics import (
    SurvivalData,
    bootstrap_metrics,
    harrell_c,
    point_metrics,
    uno_c,
)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n = 300
    risk = rng.normal(size=n)
    event_time = rng.exponential(scale=5 * np.exp(-risk))
    censor_time = rng.uniform(1, 12, size=n)
    time = np.minimum(event_time, censor_time)
    event = event_time <= censor_time
    return time, event, risk


def test_perfect_and_reversed_ranking(synthetic):
    time, event, _ = synthetic
    perfect = SurvivalData.from_arrays(time, event, -time)
    assert harrell_c(perfect) == pytest.approx(1.0)
    reversed_ = SurvivalData.from_arrays(time, event, time)
    assert harrell_c(reversed_) == pytest.approx(0.0)


def test_informative_score_beats_chance(synthetic):
    data = SurvivalData.from_arrays(*synthetic)
    assert 0.6 < harrell_c(data) < 0.9
    assert 0.6 < uno_c(data, tau=8) < 0.9


def test_point_metric_names(synthetic):
    m = point_metrics(SurvivalData.from_arrays(*synthetic), horizons=[2, 5], tau=8)
    assert set(m) == {"harrell_c", "uno_c_tau8", "td_auc_2", "td_auc_5"}


def test_bootstrap_ci_contains_estimate_and_is_reproducible(synthetic):
    data = SurvivalData.from_arrays(*synthetic)
    a = bootstrap_metrics(data, horizons=[2, 5], tau=8, n_boot=50, seed=1)
    b = bootstrap_metrics(data, horizons=[2, 5], tau=8, n_boot=50, seed=1)
    assert ((a["ci_lower"] <= a["estimate"]) & (a["estimate"] <= a["ci_upper"])).all()
    assert (a["n_boot_ok"] == 50).all()
    assert a.equals(b)


@pytest.mark.parametrize(
    "time, event, risk, match",
    [
        ([1, 2], [1, 0], [0.1], "equal length"),
        ([1, np.nan], [1, 0], [0.1, 0.2], "missing"),
        ([0, 2], [1, 0], [0.1, 0.2], "> 0"),
        ([1, 2], [0, 0], [0.1, 0.2], "no events"),
    ],
)
def test_invalid_inputs(time, event, risk, match):
    with pytest.raises(ValueError, match=match):
        SurvivalData.from_arrays(time, event, risk)


def test_competing_risk_code_rejected():
    with pytest.raises(ValueError, match="0/1"):
        SurvivalData.from_arrays([1, 2, 3], [1, 2, 0], [0.1, 0.2, 0.3])


def test_infinite_values_rejected():
    with pytest.raises(ValueError, match="finite"):
        SurvivalData.from_arrays([1, 2], [1, 0], [0.1, np.inf])


def test_horizon_beyond_follow_up_rejected(synthetic):
    data = SurvivalData.from_arrays(*synthetic)
    with pytest.raises(ValueError, match="within follow-up"):
        point_metrics(data, horizons=[2, 50], tau=8)


def test_separate_train_for_censoring(synthetic):
    time, event, risk = synthetic
    train = SurvivalData.from_arrays(time[:200], event[:200], risk[:200])
    test = SurvivalData.from_arrays(time[200:], event[200:], risk[200:])
    m = point_metrics(test, horizons=[2], tau=6, train=train)
    assert 0.5 < m["uno_c_tau6"] < 1.0
