# Decision log
Format: `YYYY-MM-DD — decision — reason — alternatives considered`

- 2026-11-03 — Primary Track B outcome = in-hospital mortality — only mortality outcome available in both MIMIC-IV and eICU, enabling external validation — 28-day (MIMIC only) kept as secondary.
- 2026-11-03 — No SMOTE — synthetic oversampling distorts calibrated risk — class weights / threshold tuning instead.
- 2026-11-03 — Restricted data never shown to Claude — PhysioNet DUA — develop on open demo databases.
- 2026-11-03 — Python pinned to 3.11 (`requires-python = ">=3.11,<3.12"`) and `numba>=0.61` added as an explicit lower bound — an open upper bound let the resolver backtrack to numba/llvmlite 0.36 (no 3.11 wheels, source build fails) via shap — none needed beyond the pin.
- 2026-11-03 — Python `pmsampsize` (PyPI 0.1.0) NOT installed: it requires Python ≥3.12, conflicting with the 3.11 stack — decide in Phase 2 between (a) R `pmsampsize` (reference implementation; document the call and output in reports) or (b) a small in-house implementation of Riley et al. 2019/2020 criteria checked against R output — alternatives: move whole project to 3.12 (rejected for now; CLAUDE.md specifies 3.11).
- 2026-11-03 — `.gitignore` data rule changed from `data/**/*` to `data/**` + `!data/**/` so `.gitkeep` placeholders are tracked; all data files (open, restricted, processed) remain ignored (verified with `git check-ignore`).
- 2026-11-03 — TRIPOD+AI tracker transcribed from Collins et al. BMJ 2024;385:e078378 Table 2 as short paraphrases with Track A/B status columns — official wording used at submission time.
