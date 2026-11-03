# TRIPOD+AI checklist tracker

Source: Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting
clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378
(Table 2; CC BY 4.0; correction BMJ 2024;385:q902). Items below are short paraphrases — use the
official wording / fillable checklist (supplementary table 2, tripod-statement.org) for submission.
The 13-item **TRIPOD+AI for Abstracts** checklist (item 2) is not transcribed here yet.

D = development only · E = evaluation only · D;E = both.
Status: ☐ not started · ◐ in progress · ☑ done · n/a not applicable.

| Item | Topic | D/E | Requirement (paraphrase) | Track A (PBC) | Track B (ICU) | Where addressed |
|---|---|---|---|---|---|---|
| 1 | Title | D;E | Say it develops/evaluates a multivariable prediction model; population; outcome | ☐ | ☐ | |
| 2 | Abstract | D;E | Follow TRIPOD+AI for Abstracts | ☐ | ☐ | |
| 3a | Background | D;E | Healthcare context (diagnostic/prognostic), rationale, existing models | ◐ | ◐ | docs/PLAN.md §0–1 |
| 3b | Background | D;E | Target population, intended purpose in care pathway, intended users | ◐ | ◐ | docs/PLAN.md |
| 3c | Background | D;E | Known health inequalities between sociodemographic groups | ☐ | ☐ | |
| 4 | Objectives | D;E | Objectives; development, validation or both | ◐ | ◐ | docs/PLAN.md §1.2, §2 |
| 5a | Data | D;E | Data sources per dataset, rationale, representativeness | ◐ | ◐ | docs/PLAN.md §1.1, §2.1 |
| 5b | Data | D;E | Dates of accrual and end of follow-up | ☐ | ☐ | |
| 6a | Participants | D;E | Setting, number and location of centres | ☐ | ☐ | |
| 6b | Participants | D;E | Eligibility criteria | ◐ | ◐ | PBC: trial n=312 primary (DECISIONS) |
| 6c | Participants | D;E | Treatments received and how handled | ◐ | ☐ | Treatment arm not a predictor (DECISIONS P2) |
| 7 | Data preparation | D;E | Pre-processing and quality checks; similar across groups? | ◐ | ☐ | `pbc/load.py`, DECISIONS |
| 8a | Outcome | D;E | Outcome definition, horizon, how/when assessed, rationale, consistency across groups | ◐ | ◐ | PBC: death, CL censored (DECISIONS) |
| 8b | Outcome | D;E | Qualifications of outcome assessors if subjective | n/a | ☐ | |
| 8c | Outcome | D;E | Blinding of outcome assessment | ☐ | ☐ | |
| 9a | Predictors | D | Choice and pre-selection of candidate predictors | ☑ | ◐ | pbc/models.py docstring; DECISIONS P2 |
| 9b | Predictors | D;E | Define predictors, how/when measured, blinding | ◐ | ◐ | pbc/models.py; configs/canonical_schema.yaml |
| 9c | Predictors | D;E | Qualifications of predictor assessors if subjective | ☐ | ☐ | PBC: edema, histologic stage |
| 10 | Sample size | D;E | How study size was arrived at; justification/calculation | ☑ | ☐ | reports/pbc_report.md §Sample size; evaluation/sample_size.py |
| 11 | Missing data | D;E | Handling of missing data; reasons for omissions | ☑ | ☐ | survival/preprocessing.py; reports/pbc/missingness.md |
| 12a | Analytical methods | D | How data were used/partitioned | ☑ | ☐ | reports/pbc_report.md §Methods; survival/cv.py |
| 12b | Analytical methods | D | Predictor handling: functional form, transforms, scaling | ☑ | ☐ | pbc/models.py (log transforms, scaling in fold) |
| 12c | Analytical methods | D | Model type, rationale, building steps, tuning, internal validation | ☑ | ☐ | survival/estimators.py; DECISIONS P2 |
| 12d | Analytical methods | D;E | Heterogeneity across clusters | n/a | ☐ | |
| 12e | Analytical methods | D;E | All performance measures/plots and rationale; model comparison | ☑ | ☐ | evaluation/{survival_metrics,calibration,dca}.py |
| 12f | Analytical methods | E | Model updating (e.g. recalibration) | ☐ | ☐ | |
| 12g | Analytical methods | E | How predictions were calculated for evaluation | ◐ | ☐ | `pbc/mayo_score.py` (apparent performance; DECISIONS) |
| 13 | Class imbalance | D;E | Imbalance methods, why, and recalibration | n/a | ◐ | DECISIONS: no SMOTE |
| 14 | Fairness | D;E | Approaches to model fairness and rationale | ☐ | ☐ | |
| 15 | Model output | D | Output type; thresholds and how chosen | ☑ | ☐ | 5-year risk of death; no classification threshold (DCA range) |
| 16 | Training vs evaluation | D;E | Differences between development and evaluation data | ☐ | ☐ | |
| 17 | Ethical approval | D;E | Ethics committee / consent or waiver | ☐ | ☐ | open/credentialed data |
| 18a | Funding | D;E | Funding source and role | ☐ | ☐ | |
| 18b | Conflicts of interest | D;E | Conflicts for all authors | ☐ | ☐ | |
| 18c | Protocol | D;E | Where protocol is accessible, or none | ◐ | ◐ | docs/PLAN.md |
| 18d | Registration | D;E | Registration details, or not registered | ☐ | ☐ | |
| 18e | Data sharing | D;E | Availability of study data | ◐ | ☐ | UCI 878 (CC BY 4.0) |
| 18f | Code sharing | D;E | Availability of analytical code | ◐ | ◐ | this repository |
| 19 | Patient & public involvement | D;E | PPI details or none | ☐ | ☐ | |
| 20a | Participants | D;E | Participant flow, n with/without outcome, follow-up summary | ◐ | ☐ | reports/pbc/table1_by_status_trial312.md, mayo_score_metrics.md |
| 20b | Participants | D;E | Characteristics overall/per source, events, follow-up, missingness; group differences | ◐ | ☐ | reports/pbc/ |
| 20c | Participants | E | Evaluation vs development predictor/outcome distributions | ☐ | ☐ | |
| 21 | Model development | D;E | n participants and events per analysis | ☑ | ☐ | reports/pbc/cv_fold_metrics.csv (n, events per fold) |
| 22 | Model specification | D | Full model (formula/code/object/API) and access terms | ◐ | ☐ | Mayo refit coefficients in report; ML models as code |
| 23a | Model performance | D;E | Performance with CIs, incl. key subgroups; plots | ☑ | ☐ | reports/pbc_report.md (CIs; no subgroups yet) |
| 23b | Model performance | D;E | Heterogeneity across clusters | n/a | ☐ | |
| 24 | Model updating | E | Results of any updating | ☐ | ☐ | |
| 25 | Interpretation | D;E | Overall interpretation incl. fairness vs objectives/prior work | ☐ | ☐ | |
| 26 | Limitations | D;E | Limitations and effects on bias, uncertainty, generalisability | ◐ | ☐ | reports/pbc_report.md §Limitations |
| 27a | Usability | D | Handling poor-quality/unavailable inputs at implementation | ☐ | ☐ | |
| 27b | Usability | D | User interaction and required expertise | ☐ | ☐ | |
| 27c | Usability | D;E | Next steps; applicability and generalisability | ☐ | ☐ | |
