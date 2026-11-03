import math

import pytest

from cirrhosis_ml.evaluation.sample_size import (
    max_r2_cs_survival,
    r2_cs_from_lr,
    survival_sample_size,
)


def test_riley_bmj_2020_survival_example():
    # Riley et al. BMJ 2020;368:m441 worked example -> 5143 participants, ~692 events, EPP 23.07
    res = survival_sample_size(r2_cs=0.051, parameters=30, rate=0.065, timepoint=2,
                               mean_followup=2.07)
    assert res.n_required == 5143
    assert res.events_required == pytest.approx(692, abs=1)
    assert res.events_per_parameter == pytest.approx(23.07, abs=0.01)
    assert res.n_criterion1 >= res.n_criterion2


def test_criterion1_closed_form():
    res = survival_sample_size(0.2, 10, rate=0.1, timepoint=5, mean_followup=5)
    expected = math.ceil(10 / (-0.1 * math.log(1 - 0.2 / 0.9)))
    assert res.n_criterion1 == expected


def test_max_r2_increases_with_event_burden():
    assert max_r2_cs_survival(0.2, 5) > max_r2_cs_survival(0.02, 5)


def test_r2_from_lr():
    assert r2_cs_from_lr(199.2, 312) == pytest.approx(1 - math.exp(-199.2 / 312))


def test_cuminc_ci_brackets_estimate():
    res = survival_sample_size(0.1, 5, rate=0.07, timepoint=5, mean_followup=5.5)
    lo, hi = res.cuminc_ci
    assert lo < res.cuminc_at_timepoint < hi


@pytest.mark.parametrize("kwargs", [{"parameters": 0}, {"r2_cs": 0.99}, {"rate": -1}])
def test_invalid_inputs(kwargs):
    base = {"r2_cs": 0.1, "parameters": 5, "rate": 0.07, "timepoint": 5, "mean_followup": 5}
    with pytest.raises(ValueError):
        survival_sample_size(**{**base, **kwargs})
