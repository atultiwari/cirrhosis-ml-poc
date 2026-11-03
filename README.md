# cirrhosis-ml-poc

Proof-of-concept ML prediction work ahead of an AI study in liver cirrhosis & portal hypertension
at the Dept of Gastroenterology, RNT Medical College, Udaipur.

| Track | Data | Question |
|---|---|---|
| **A** | Mayo Clinic PBC (UCI 878, open, CC BY 4.0) | Reusable survival-modelling pipeline; ML vs Mayo risk score |
| **B** | MIMIC-IV (dev + temporal) → eICU-CRD (external) — credentialed | ICU cirrhosis in-hospital / 28-day mortality and AKI vs MELD-Na, MELD 3.0, CLIF-C |

- **Plan:** `docs/PLAN.md`
- **Start in Claude Code:** `docs/STARTER_PROMPT.md`
- **Rules for Claude (incl. data governance):** `CLAUDE.md`
- **PhysioNet access checklist:** `docs/PHYSIONET_ACCESS.md`

⚠️ Credentialed MIMIC-IV / eICU data stays in `data/restricted/`, is never committed, and is never shown to any AI assistant.

## Quick start
```bash
uv sync                                         # Python 3.11 env from uv.lock
uv run pre-commit install                       # ruff + "no data files" commit guard
uv run pytest                                   # tests (open PBC data only)
uv run python -m cirrhosis_ml.pbc.run_all       # Track A: reproduces reports/pbc/
```

## Data & attribution
- Mayo Clinic PBC data: UCI Machine Learning Repository, *Cirrhosis Patient Survival Prediction*
  (id 878), CC BY 4.0 — downloaded at run time, never committed.
- MIMIC-IV and eICU-CRD are credentialed PhysioNet datasets and are **not** in this repository.

Research proof-of-concept only — not a clinical decision tool.
