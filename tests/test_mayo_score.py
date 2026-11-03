import numpy as np
import pytest

from cirrhosis_ml.pbc.mayo_score import (
    MAYO_INITIAL_COEFFICIENTS,
    mayo_risk_score,
    mayo_score_from_pbc,
    mayo_survival_probability,
)


def test_dickson_example_1_low_risk():
    # Dickson 1989 p6: bili 0.5, albumin 4.5, age 52, PT 10.1, no edema -> R = 3.12, S(5y) = 0.96
    r = mayo_risk_score(52, 0.5, 4.5, 10.1, 0.0)
    assert r == pytest.approx(3.12, abs=0.005)
    assert mayo_survival_probability(r, 5) == pytest.approx(0.96, abs=0.005)


def test_dickson_example_2_high_risk():
    # Dickson 1989 p6: bili 13.9, albumin 2.8, age 52, PT 13.8, edema 0.5 -> R = 8.39, S(1y) = 0.43
    r = mayo_risk_score(52, 13.9, 2.8, 13.8, 0.5)
    assert r == pytest.approx(8.39, abs=0.005)
    assert mayo_survival_probability(r, 1) == pytest.approx(0.43, abs=0.005)


def test_vectorised_and_monotone_in_bilirubin():
    r = mayo_risk_score([50, 50], [1.0, 5.0], [3.5, 3.5], [10.5, 10.5], [0, 0])
    assert r.shape == (2,)
    assert r[1] > r[0]


def test_albumin_protective():
    assert mayo_risk_score(50, 1, 4.0, 10.5, 0) < mayo_risk_score(50, 1, 3.0, 10.5, 0)


def test_initial_coefficients_differ():
    a = mayo_risk_score(52, 0.5, 4.5, 10.1, 0.0)
    b = mayo_risk_score(52, 0.5, 4.5, 10.1, 0.0, coefficients=MAYO_INITIAL_COEFFICIENTS)
    assert a != pytest.approx(b)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bilirubin_mg_dl": 0.0}, "bilirubin"),
        ({"albumin_g_dl": -1.0}, "albumin"),
        ({"edema": 0.3}, "edema"),
        ({"prothrombin_s": np.nan}, "missing"),
    ],
)
def test_invalid_inputs_raise(kwargs, match):
    base = {"age_years": 50, "bilirubin_mg_dl": 1.0, "albumin_g_dl": 3.5, "prothrombin_s": 10.5,
            "edema": 0.0}
    with pytest.raises(ValueError, match=match):
        mayo_risk_score(**{**base, **kwargs})


def test_survival_years_out_of_range():
    with pytest.raises(ValueError, match="years"):
        mayo_survival_probability(5.0, 10)


def test_initial_model_reproduces_dickson_table3(pbc):
    # Table 3 (Mayo study, n=312) R percentiles 10th/50th/90th = 4.03/5.24/7.67; computed with
    # the initial (n=312) model, which was fitted before the final model.
    trial = pbc[pbc["is_trial"]]
    r = mayo_score_from_pbc(trial, coefficients=MAYO_INITIAL_COEFFICIENTS)
    np.testing.assert_allclose(np.percentile(r, [10, 50, 90]), [4.03, 5.24, 7.67], atol=0.06)


def test_final_model_mean_matches_centring_constant(pbc):
    # p6: 5.07 is the mean R of the combined 418 patients (2 lack prothrombin time in UCI 878)
    complete = pbc[pbc["Prothrombin"].notna()]
    assert mayo_score_from_pbc(complete).mean() == pytest.approx(5.07, abs=0.05)
