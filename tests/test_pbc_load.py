import numpy as np
import pandas as pd
import pytest

from cirrhosis_ml.pbc.load import DAYS_PER_YEAR, N_TOTAL, N_TRIAL, prepare_pbc


def _raw_row(**overrides):
    row = {
        "ID": "1", "N_Days": "730", "Status": "D", "Drug": "Placebo", "Age": "18262",
        "Sex": "F", "Ascites": "N", "Hepatomegaly": "Y", "Spiders": "N", "Edema": "S",
        "Bilirubin": "1.2", "Cholesterol": "NaNN", "Albumin": "3.5", "Copper": "50",
        "Alk_Phos": "1000", "SGOT": "100", "Tryglicerides": "NaNN", "Platelets": "200",
        "Prothrombin": "10.5", "Stage": "3",
    }
    return {**row, **overrides}


def test_row_counts(pbc):
    assert len(pbc) == N_TOTAL
    assert pbc["is_trial"].sum() == N_TRIAL
    assert pbc["ID"].is_unique


def test_status_coding(pbc):
    assert pbc["event_cr"].value_counts().to_dict() == {0: 232, 1: 161, 2: 25}
    assert pbc["event_death"].sum() == 161
    # transplanted patients are censored in the primary analysis
    assert (pbc.loc[pbc["event_cr"] == 2, "event_death"] == 0).all()


def test_unit_conversion(pbc):
    np.testing.assert_allclose(pbc["age_years"], pbc["Age"] / DAYS_PER_YEAR)
    assert pbc["age_years"].between(20, 90).all()
    np.testing.assert_allclose(pbc["time_years"], pbc["N_Days"] / DAYS_PER_YEAR)


def test_missing_tokens_become_nan(pbc):
    assert pbc["Cholesterol"].dtype.kind == "f"
    # known trial-cohort missingness: copper 2, cholesterol 28, triglycerides 30
    trial = pbc[pbc["is_trial"]]
    assert trial[["Copper", "Cholesterol", "Tryglicerides"]].isna().sum().tolist() == [2, 28, 30]
    assert pbc["drug_dpen"].isna().sum() == N_TOTAL - N_TRIAL


def test_edema_and_binary_encoding():
    df = prepare_pbc(pd.DataFrame([_raw_row()]))
    row = df.iloc[0]
    assert row["edema_score"] == 0.5
    assert row["hepatomegaly"] == 1.0 and row["ascites"] == 0.0
    assert row["sex_male"] == 0.0 and row["drug_dpen"] == 0.0
    assert np.isnan(row["Cholesterol"])
    assert row["age_years"] == pytest.approx(18262 / DAYS_PER_YEAR)


def test_prepare_does_not_mutate_input():
    raw = pd.DataFrame([_raw_row()])
    before = raw.copy()
    prepare_pbc(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_unknown_status_raises():
    with pytest.raises(ValueError, match="Status"):
        prepare_pbc(pd.DataFrame([_raw_row(Status="X")]))


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing expected columns"):
        prepare_pbc(pd.DataFrame([_raw_row()]).drop(columns=["Bilirubin"]))


def test_trial_flag_consistency_checked():
    with pytest.raises(ValueError, match="Trial flag"):
        prepare_pbc(pd.DataFrame([_raw_row(ID="400")]))


def test_missing_mandatory_value_raises():
    with pytest.raises(ValueError, match="Mandatory"):
        prepare_pbc(pd.DataFrame([_raw_row(Status="NaNN")]))
