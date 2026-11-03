import numpy as np
import pytest

from cirrhosis_ml.evaluation.calibration import calibrate, km_risk, truncate_at
from cirrhosis_ml.evaluation.dca import decision_curves, net_benefit

HORIZON = 5.0


@pytest.fixture
def exponential_cohort():
    """Event times with known individual 5-y risk; administrative censoring at 8 y."""
    rng = np.random.default_rng(3)
    n = 4000
    lp = rng.normal(0, 1, n)
    rate = 0.05 * np.exp(lp)
    event_time = rng.exponential(1 / rate)
    time = np.minimum(event_time, 8.0)
    event = (event_time <= 8.0).astype(int)
    true_risk = 1 - np.exp(-rate * HORIZON)
    return time, event, true_risk


def test_true_risk_is_well_calibrated(exponential_cohort):
    time, event, risk = exponential_cohort
    cal = calibrate(time, event, risk, HORIZON)
    assert cal.oe_ratio == pytest.approx(1.0, abs=0.06)
    assert cal.slope == pytest.approx(1.0, abs=0.1)
    assert len(cal.table) == 10


def test_overfitted_risk_has_slope_below_one(exponential_cohort):
    time, event, risk = exponential_cohort
    cll = np.log(-np.log(1 - risk))
    too_extreme = 1 - np.exp(-np.exp(2 * (cll - cll.mean()) + cll.mean()))
    assert calibrate(time, event, too_extreme, HORIZON).slope < 0.7


def test_km_risk_no_censoring():
    time = np.array([1.0, 2.0, 6.0, 7.0])
    risk, lo, hi = km_risk(time, np.ones(4), HORIZON)
    assert risk == pytest.approx(0.5)
    assert lo <= risk <= hi


def test_truncate_at():
    t, e = truncate_at(np.array([2.0, 6.0]), np.array([1, 1]), HORIZON)
    assert t.tolist() == [2.0, 5.0] and e.tolist() == [1, 0]


def test_calibrate_rejects_non_probabilities():
    with pytest.raises(ValueError, match="probabilities"):
        calibrate([1, 2], [1, 0], [0.2, 1.5], HORIZON)


def test_net_benefit_model_beats_treat_all_at_high_threshold(exponential_cohort):
    time, event, risk = exponential_cohort
    curves = decision_curves(time, event, {"true": risk}, HORIZON, thresholds=[0.1, 0.4])
    assert curves.loc[0.4, "true"] > curves.loc[0.4, "treat_all"]
    assert (curves["treat_none"] == 0).all()


def test_treat_all_formula():
    time = np.array([1.0, 2.0, 6.0, 7.0])
    nb = net_benefit(time, np.ones(4), np.ones(4), HORIZON, thresholds=[0.2])
    assert nb[0] == pytest.approx(0.5 - 0.5 * 0.25)
