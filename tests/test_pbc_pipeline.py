from cirrhosis_ml.pbc.eda import missingness, run_eda, table_one
from cirrhosis_ml.pbc.run_all import SCORE_VARIANTS, _cohort_counts, evaluate_mayo


def test_table_one_counts(pbc):
    t = table_one(pbc[pbc["is_trial"]])
    assert t.loc["n"].to_dict() == {
        "Overall": "312", "Censored (C)": "168", "Died (D)": "125", "Transplant (CL)": "19",
    }


def test_missingness_trial_split(pbc):
    m = missingness(pbc, ("Copper",))
    assert m.loc["Copper", "Non-trial"].startswith("106 ")


def test_run_eda_writes_outputs(pbc, tmp_path):
    written = run_eda(pbc, tmp_path)
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)
    assert {p.name for p in written} >= {"km_overall.png", "km_by_stage.png", "missingness.md"}


def test_evaluate_mayo_small_bootstrap(pbc):
    trial = pbc[pbc["is_trial"]]
    metrics = evaluate_mayo(trial, n_boot=20, seed=0)
    assert len(metrics) == len(SCORE_VARIANTS) * 5
    c = metrics.loc[(next(iter(SCORE_VARIANTS)), "harrell_c"), "estimate"]
    assert 0.80 < c < 0.88  # apparent C of the Mayo score on the derivation patients
    assert _cohort_counts(trial).loc["deaths", "n"] == 125
