# Starter prompt for Claude Code

Open Terminal, `cd ~/Projects/medical-health/cirrhosis-ml-poc`, run `claude`, and paste the block below.
Suggested: start in **plan mode** (Shift+Tab) for the first message so you can approve the plan before files change.

---

```
You are joining an existing research project. Before doing anything:
1. Read CLAUDE.md completely — especially the 🔒 data-governance rules. Treat them as hard constraints
   for the whole session: never read or print anything under data/restricted/ or data derived from it.
2. Read docs/PLAN.md, docs/DECISIONS.md, data/README.md and configs/*.yaml.

Then do Phase 0 and Phase 1 (Track A start):

PHASE 0 — project scaffolding
- Initialise a uv project (Python 3.11) with a src layout: package `cirrhosis_ml` under src/.
- Add dependencies: pandas, polars, duckdb, numpy, scipy, scikit-learn, scikit-survival, lifelines,
  xgboost, lightgbm, shap, statsmodels, matplotlib, pyyaml, ucimlrepo; dev: pytest, ruff, ipykernel.
  Check whether a Python `pmsampsize` package exists and works; if not, note it in DECISIONS.md.
- Set up ruff + pytest config in pyproject.toml, and a minimal tests/test_smoke.py.
- Create module skeletons: cirrhosis_ml/{config.py, io.py, survival/, models/, evaluation/, pbc/, mimic/, eicu/, reporting/}.
- Populate docs/TRIPOD_AI_checklist.md with the TRIPOD+AI items (fetch the item list from
  tripod-statement.org / the BMJ 2024 paper if you can; otherwise create the structure and flag it).
- git init + first commit (data and models are already git-ignored — double-check .gitignore).

PHASE 1 — Track A: Mayo PBC data, EDA and Mayo risk score
- Write cirrhosis_ml/pbc/load.py: fetch UCI dataset id 878 via ucimlrepo, cache to data/open/pbc/,
  convert Age from days to years, encode Status (D=event; C and CL censored for primary analysis;
  keep a competing-risk coding too), flag trial (n=312) vs non-trial patients.
- Write cirrhosis_ml/pbc/eda.py that saves to reports/: table 1 by status, missingness summary,
  Kaplan–Meier overall and by Stage.
- Implement the Mayo PBC risk score in cirrhosis_ml/pbc/mayo_score.py. Verify the coefficients against
  Dickson et al., Hepatology 1989 (search the web for the original formula) and cite the source in a
  docstring and in DECISIONS.md. Evaluate it on the 312 trial patients: Harrell's and Uno's C,
  time-dependent AUC at 2/5/10 years.
- Add tests for loading (row counts, unit conversion, status coding) and the score function.
- Add a single entry point `uv run python -m cirrhosis_ml.pbc.run_all` that reproduces everything so far.

Working style:
- Show me a short plan first, then implement step by step; run tests after each module.
- Log every methodological choice in docs/DECISIONS.md.
- At the end, update the Phase checklist in docs/PLAN.md and summarise what is done, what's next
  (Phase 2: models + nested CV + calibration + DCA + SHAP), and anything you need from me.
```

---

## Follow-up prompts (use in order, one per session or as phases complete)

**Phase 2 — Track A models**
```
Continue with PLAN.md Phase P2 (Track A models). Build the reusable survival module in
src/cirrhosis_ml/survival/: Cox PH (with PH checks), penalised Cox, Random Survival Forest and
gradient-boosted survival; repeated nested CV (5×10 outer, inner tuning); metrics = Harrell/Uno C,
time-dependent AUC at 2/5/10 y, integrated Brier, 5-year calibration (decile plot + slope), DCA at 5 y,
all with bootstrap CIs; SHAP for tree models. Compare against the Mayo score. Sensitivity analyses:
all 418 patients with MICE inside folds; competing-risk with transplant. Produce reports/pbc_report.md
with figures. Keep everything leakage-free and logged in DECISIONS.md.
```

**Phase 4 — Track B on demo data**
```
Start Track B (PLAN.md section 2) using ONLY data/open/mimic_iv_demo and data/open/eicu_demo.
Remember the CLAUDE.md data rules: scripts must take DATA_ROOT as input, and I will run them on
data/restricted myself. Write DuckDB SQL in sql/mimic_iv/ (numbered) to build: cirrhosis cohort
(configs/icd_codes.yaml — first verify the code list against published MIMIC-IV cirrhosis studies
and record the source), first-24h features, outcomes (in-hospital, 28-day, 90-day mortality;
KDIGO AKI ≥2 between 24h and 7d), and scores (MELD, MELD-Na, MELD 3.0, SOFA, CLIF-C OF/ACLF where
derivable — verify formulas from the original papers). Map every feature to
configs/canonical_schema.yaml. Mirror for eICU. Add tests that check feature timestamps never exceed
the 24-h window. Output a flow diagram of counts (suppress n<10). Do not train models yet.
```

**Phase 6–7 — models + validation (code only; Atul runs on full data)**
```
Write the Track B modelling and evaluation scripts (PLAN.md 2.6–2.7): temporal split on
anchor_year_group, LR with splines, LASSO-LR, RF, XGBoost, LightGBM (class weights, no SMOTE),
bootstrap CIs, DeLong vs MELD-Na/MELD 3.0/CLIF-C, calibration, DCA, SHAP, subgroups, eICU external
validation with intercept recalibration. Test end-to-end on the demo data only. Outputs must be
aggregate-only. Then give me the exact commands to run on data/restricted and the list of output
files I can safely share back with you.
```
