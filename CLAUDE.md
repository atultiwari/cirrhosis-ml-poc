# CLAUDE.md — Cirrhosis ML PoC (RNT Medical College, Udaipur)

## What this project is
Proof-of-concept ML prediction work that precedes a prospective AI study in liver cirrhosis /
portal hypertension at the Dept of Gastroenterology, RNT Medical College, Udaipur.
Two tracks (full detail in `docs/PLAN.md`):
- **Track A — PBC survival pipeline** on the open Mayo Clinic PBC dataset (UCI 878).
  Purpose: build and validate a reusable, TRIPOD+AI-grade survival-modelling pipeline.
- **Track B — ICU cirrhosis outcome models** on MIMIC-IV (development + temporal validation)
  and eICU-CRD (external validation): in-hospital / 28-day mortality and AKI.
Both tracks must write into a **canonical feature schema** (`configs/canonical_schema.yaml`) so the
same pipeline can later run on RNT's own registry data.

The owner (Atul) is a pathologist with an MS in ML/AI. He builds via Claude Code, so explain
decisions briefly but do not over-explain basic ML.

## 🔒 HARD DATA-GOVERNANCE RULES (non-negotiable)
1. **Never read, open, cat, head, sample, print, or summarise any file under `data/restricted/`**
   or any file in `data/processed/` derived from it. Do not run commands whose stdout would show
   patient-level rows from those sources. This is required by the PhysioNet Credentialed Data Use
   Agreement (no sharing with third-party online services, which includes this AI assistant).
2. Develop and debug all SQL/Python against `data/open/mimic_iv_demo/` and `data/open/eicu_demo/`.
   Scripts must take the data root as a CLI argument / env var (`DATA_ROOT`) so Atul can point
   them at `data/restricted/...` and run them himself.
3. Scripts that run on restricted data must only emit **aggregate** outputs to stdout/files:
   model metrics, calibration tables, figures, cohort counts. Suppress any cell with n < 10.
4. Never commit data or trained models fitted on restricted data (see `.gitignore`).
5. If unsure whether something touches restricted data — stop and ask.

## Methodological non-negotiables
- Split first (patient-level, temporal where applicable), then impute/scale/select **inside**
  the training fold only (sklearn `Pipeline` / `ColumnTransformer`). No leakage.
- Predictors must be available at the prediction time point (e.g. first 24 h of ICU stay).
  Discharge ICD codes are NOT baseline predictors unless explicitly flagged as a sensitivity analysis.
- No SMOTE / synthetic oversampling (it miscalibrates risk). Use class weights if needed.
- Always report: discrimination (AUROC / C-index with 95% bootstrap CI), calibration
  (intercept, slope, plot), Brier score, decision-curve analysis.
- Always benchmark against clinical scores (MELD, MELD-Na, MELD 3.0, CTP where computable,
  SOFA, CLIF-C OF / ACLF where computable; Mayo risk score for PBC) and plain logistic / Cox regression.
- Fix random seeds; log package versions; every result reproducible from a single command.
- Follow TRIPOD+AI for reporting; keep `docs/TRIPOD_AI_checklist.md` updated as work proceeds.

## Stack & conventions
- Python 3.11, `uv` for env/deps (`uv sync`, `uv run`). DuckDB for MIMIC/eICU CSV querying (no BigQuery).
- Libraries: pandas/polars, duckdb, scikit-learn, scikit-survival, lifelines, xgboost, lightgbm,
  shap, statsmodels, matplotlib, ucimlrepo, pmsampsize (Python port; if unavailable, document R usage).
- Package code in `src/cirrhosis_ml/`; thin CLI entry points (`python -m cirrhosis_ml.<module>`).
- SQL lives in `sql/mimic_iv/` and `sql/eicu/`, one file per derived table, numbered in run order.
- Config in YAML under `configs/` (outcome definitions, feature lists, ICD code lists).
- Tests with pytest in `tests/`, using the open demo data / PBC only.
- Notebooks are for exploration only; anything that produces a reported number must be a script.
- Write a short entry in `docs/DECISIONS.md` for every non-trivial methodological choice.

## Useful commands (create as the project grows)
- `uv run python -m cirrhosis_ml.pbc.run_all` — Track A end to end
- `DATA_ROOT=data/open/mimic_iv_demo uv run python -m cirrhosis_ml.mimic.build_cohort`
- `uv run pytest`
