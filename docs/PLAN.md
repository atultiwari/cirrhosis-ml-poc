# Cirrhosis ML Proof-of-Concept — Detailed Plan

**Owner:** Dr Atul Tiwari · **Clinical home:** Dept of Gastroenterology, RNT Medical College, Udaipur
**Status:** Planning → Phase 0 · **Last updated:** 2026-11-03

---

## 0. Why this PoC exists
The eventual goal is an AI/ML study at RNT on liver cirrhosis, portal hypertension and their
complications (severity, complication prediction, mortality). Before collecting local data we want:
1. A **validated, reusable modelling pipeline** (Track A).
2. A **credible pilot result on real-world cirrhosis data** with temporal + external validation (Track B),
   to support the IEC submission, convince the GE department, and serve as the comparator model that
   will later be externally validated on RNT patients.
3. A **canonical feature schema** that RNT's registry/eCRF will map onto, so the same code runs on
   Indian data with minimal change.

Non-goal: a clinically deployable tool. These are research PoCs.

---

## 1. Track A — Survival-modelling pipeline on the Mayo PBC dataset

### 1.1 Data
- Source: UCI ML Repository id **878** "Cirrhosis Patient Survival Prediction" (Mayo Clinic PBC
  trial, 1974–1984), **418 patients, 17 features, CC BY 4.0**. Fetch with `ucimlrepo`.
- 312 randomised trial patients (D-penicillamine vs placebo) with near-complete data; 106 non-trial
  patients with many missing labs.
- Time: `N_Days`. Status: `C` (censored), `CL` (censored at liver transplant), `D` (death).
- Note: `Age` is in **days** → convert to years.
- Optional richer version: `pbcseq` (longitudinal, 312 pts) via R `survival` package / lib.stat.cmu.edu.

### 1.2 Questions
- Primary: predict overall survival (time-to-death); report risk at 2, 5 and 10 years.
- Secondary: compare ML survival models vs the classic **Mayo risk score** (Dickson et al., Hepatology
  1989; R = 0.871·ln(bilirubin) − 2.53·ln(albumin) + 0.039·age + 2.38·ln(PT) + 0.859·edema —
  **verify coefficients against the original paper before use**).

### 1.3 Analysis steps
1. **EDA** → table 1 (by status), missingness map, KM curves overall and by stage.
2. **Analysis populations:** primary = 312 trial patients; sensitivity = all 418 with multiple imputation.
3. **Transplant handling:** primary = treat `CL` as censored; sensitivity = competing-risk
   (Aalen–Johansen CIF, Fine–Gray) with transplant as competing event.
4. **Missing data:** MICE (`IterativeImputer`) inside the training fold; report complete-case sensitivity.
5. **Models:**
   - Mayo risk score (fixed formula, no fitting) → recalibrated Cox on the score
   - Cox PH (lifelines; check PH assumption with Schoenfeld residuals)
   - Penalised Cox (elastic net, scikit-survival `CoxnetSurvivalAnalysis`)
   - Random Survival Forest (scikit-survival)
   - Gradient-boosted survival (`GradientBoostingSurvivalAnalysis` or XGBoost `survival:cox`)
   - Optional: DeepSurv (pycox) — only if time permits; expect no gain at n≈300.
6. **Validation:** repeated nested CV (outer 5-fold × 10 repeats, inner CV for tuning) + optimism-corrected
   bootstrap (Harrell) for Cox/score models.
7. **Metrics:** Harrell's C and Uno's C; time-dependent AUC at 2/5/10 y; integrated Brier score;
   calibration at 5 y (observed KM vs predicted in deciles + calibration slope); DCA at 5 y.
8. **Explainability:** SHAP for tree models (survival SHAP or SHAP on risk score); permutation importance;
   partial dependence for bilirubin, albumin, PT, age.
9. **Sample-size sanity check:** `pmsampsize` for a survival outcome — expect it to show the dataset is
   small for many predictors; document this honestly.

### 1.4 Deliverables (Track A)
- `src/cirrhosis_ml/survival/` — reusable module (fit/evaluate/calibrate/DCA/report) that Track B and
  the RNT study will reuse for 90-day and 1-year outcomes.
- `reports/pbc_report.html` (or .md + figures) with TRIPOD+AI-style sections.
- Optional demo: a small Streamlit/FastAPI calculator for the PBC model (fine to share — open data).
- Teaching asset: the notebook/report doubles as workshop material for VRL courses.

**Exit criteria:** pipeline runs end-to-end from one command; tests pass; ML models compared to the
Mayo score with CIs; one-page summary of what did/didn't beat the classic score.

---

## 2. Track B — ICU cirrhosis outcome models (MIMIC-IV → eICU-CRD)

### 2.1 Data access (start immediately, runs in parallel with Track A)
- PhysioNet account → CITI "Data or Specimens Only Research" course → credentialing request with
  reference (supervisor/colleague) → sign DUA for **MIMIC-IV** (v3.x) and **eICU-CRD** (v2.0).
  Typical turnaround 1–2 weeks. Checklist in `docs/PHYSIONET_ACCESS.md`.
- Meanwhile build everything on the **open demo databases** (same schema):
  MIMIC-IV Clinical Database Demo (100 patients) and eICU-CRD Demo. Verify current licence on PhysioNet.
- Storage: download CSVs to `data/restricted/...` on the local Mac (encrypted disk). Query with DuckDB.
- 🔒 **Claude must never see restricted data** — see CLAUDE.md. Atul runs the final scripts locally.

### 2.2 Cohort (MIMIC-IV)
- Adults ≥18 y, **first ICU stay** of first hospital admission with a cirrhosis diagnosis.
- Cirrhosis by ICD codes (starting list — **verify against code lists used in published MIMIC
  cirrhosis papers**, put final list in `configs/icd_codes.yaml`):
  - ICD-9: 571.2, 571.5, 571.6
  - ICD-10: K70.30, K70.31, K70.40, K70.41, K74.3, K74.4, K74.5, K74.60, K74.69, K71.7
  - Supporting complication codes (descriptive / sensitivity only): portal HTN (572.3 / K76.6),
    varices (456.0–456.21 / I85.x), hepatic encephalopathy/failure (572.2 / K72.x),
    SBP (567.23 / K65.2), HRS (572.4 / K76.7)
- Exclusions: ICU LOS < 24 h (for 24-h prediction window), liver transplant during admission,
  HCC (C22.0 / 155.0) in primary analysis (include in sensitivity), missing outcome.
- Report a STROBE/TRIPOD flow diagram with counts.

### 2.3 Prediction time point, predictors
- **Prediction time = 24 h after ICU admission**; predictors = first-24 h values (worst or first,
  pre-specified per variable).
- Candidate predictors (map each to the canonical schema):
  - Demographics: age, sex
  - Liver: bilirubin, INR, albumin, ALT, AST, ALP, platelets
  - Renal/electrolytes: creatinine, BUN, sodium, potassium, bicarbonate
  - Haematology/inflammation: Hb, WBC, (neutrophil/lymphocyte if available)
  - Perfusion/acid–base: lactate, pH, MAP min, HR max, RR max, SpO2 min, temperature
  - Neuro: GCS min
  - Organ support in 24 h: vasopressors, mechanical ventilation, RRT
  - Aetiology flag (alcohol-related vs other) — ICD-derived → flag as sensitivity only
- Avoid ICD-coded complications as predictors in primary models (coded at discharge → leakage).

### 2.4 Outcomes
| Outcome | Definition | Available in |
|---|---|---|
| **Primary: in-hospital mortality** | death before hospital discharge | MIMIC-IV ✅, eICU ✅ |
| 28-day mortality | death ≤28 d from ICU admission (MIMIC `dod` incl. out-of-hospital up to 1 y) | MIMIC-IV only |
| 90-day mortality | as above | MIMIC-IV only |
| **AKI** (secondary model) | KDIGO stage ≥2 by creatinine (± urine output) between 24 h and 7 d, among patients **without** stage ≥2 AKI in first 24 h | MIMIC-IV, eICU |
| HRS (exploratory) | ICD HRS code + KDIGO AKI + terlipressin/vasopressor+albumin — document limitations | MIMIC-IV |

Primary = in-hospital mortality because it is the only mortality outcome that can be externally
validated in eICU.

### 2.5 Clinical comparators (compute from first-24 h data)
MELD, MELD-Na, **MELD 3.0** (Kim et al., Gastroenterology 2021 — implement from paper),
Child-Pugh (approximate: ascites/HE not reliably available → document), SOFA (mimic-code concept),
CLIF-C OF and CLIF-C ACLF score (where all components derivable), plus logistic regression on the
same predictors.

### 2.6 Models
Logistic regression (restricted cubic splines for key continuous variables), LASSO-LR,
Random Forest, XGBoost, LightGBM. Class weights allowed; **no SMOTE**. Hyperparameters tuned with
inner CV on the development set only. Optional: survival models (from Track A module) for 28/90-day.

### 2.7 Validation design
1. **Development:** MIMIC-IV `anchor_year_group` 2008–2016.
2. **Temporal validation:** MIMIC-IV 2017–2019 (and 2020–2022 if present in the version used).
3. **External validation:** eICU-CRD (multi-centre US, 2014–2015) — in-hospital mortality & AKI only.
4. Metrics with 1000× bootstrap 95% CIs: AUROC (DeLong comparisons vs MELD-Na/MELD 3.0/CLIF-C),
   AUPRC, calibration intercept/slope + plots, Brier, DCA. Recalibration (intercept-only) analysis on eICU.
5. Subgroups: sex, alcohol vs non-alcohol, age bands, vasopressor use.
6. Explainability: SHAP summary + dependence plots for the final model.
7. Sample size: `pmsampsize` for binary outcome using anticipated prevalence (~25–35% in-hospital
   mortality in ICU cirrhosis) and C-statistic; report alongside achieved n.

### 2.8 Deliverables (Track B)
- `sql/mimic_iv/*.sql`, `sql/eicu/*.sql` — cohort + feature + outcome derivations (DuckDB dialect).
- `src/cirrhosis_ml/{mimic,eicu,models,evaluation}` — all runnable on demo data and (by Atul) on full data.
- `reports/trackB_results.md` + figures: flow diagram, table 1, ROC, calibration, DCA, SHAP.
- Draft manuscript skeleton following TRIPOD+AI (`docs/manuscript_outline.md`).
- PoC summary (2 pages) for IEC/GE department.

**Exit criteria:** cohort reproducible on demo data; Atul has run the full pipeline locally; model
compared to MELD-Na / MELD 3.0 / CLIF-C with CIs in temporal and external sets; honest discussion of
transportability limits (US ICU, alcohol/HCV-dominant, few HBV patients) for Indian use.

---

## 3. Canonical feature schema (bridge to the RNT study)
`configs/canonical_schema.yaml` defines, per variable: name, unit (SI + conventional), plausible range,
time window, source mapping (PBC column / MIMIC itemid / eICU labname / RNT eCRF field).
The RNT prospective eCRF should be designed from this file (plus FibroScan LSM/SSM, endoscopy
grading, Baveno VII and ICA-AKI definitions that the public data lack).

---

## 4. Timeline (indicative)
| Week | Track A | Track B |
|---|---|---|
| 1 | Phase 0 setup; PBC EDA + Mayo score | Start PhysioNet credentialing; download demos |
| 2 | Models + nested CV + metrics | Cohort SQL on MIMIC demo; canonical schema |
| 3 | Calibration, DCA, SHAP, report | Features, outcomes, scores (MELD 3.0, SOFA, CLIF-C) on demo |
| 4 | Refactor survival module; tests | Credentials arrive → Atul runs full MIMIC build locally |
| 5–6 | (Optional calculator demo) | Model development + temporal validation |
| 7 | — | eICU external validation |
| 8 | — | Report, TRIPOD+AI checklist, PoC summary for IEC |

---

## 5. Phase checklist
- [ ] **P0** uv project, pyproject, folder structure, pytest, pre-commit (ruff), `configs/` stubs
- [ ] **P1** Track A data load + EDA + Mayo score
- [ ] **P2** Track A models, validation, metrics, report
- [ ] **P3** PhysioNet credentialing (Atul) — see `docs/PHYSIONET_ACCESS.md`
- [ ] **P4** Track B cohort/features/outcomes/scores on MIMIC demo + eICU demo
- [ ] **P5** Atul runs Track B on full MIMIC-IV locally → aggregate outputs only
- [ ] **P6** Models + temporal validation
- [ ] **P7** eICU external validation
- [ ] **P8** Reports, TRIPOD+AI checklist, IEC-facing PoC summary, RNT eCRF draft from schema

---

## 6. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Restricted data accidentally shown to Claude | CLAUDE.md rules; demo-first dev; aggregate-only outputs; Atul runs restricted jobs |
| ICD-based cohort misclassification | Use validated code lists; sensitivity with stricter definition (≥2 codes or + complication code) |
| Leakage via discharge codes / post-24 h data | Strict prediction window; feature-timestamp checks in tests |
| ML no better than MELD/CLIF-C | Still publishable as a rigorous negative/validation result; report honestly |
| Poor transportability to India | Planned external validation on RNT cohort is the whole point of the next study |
| Small PBC sample | Treat Track A as pipeline validation, not clinical claim |

---

## 7. Key references
See `docs/REFERENCES.md`.
