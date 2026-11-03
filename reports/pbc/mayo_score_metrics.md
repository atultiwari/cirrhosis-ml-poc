# Mayo PBC risk score — discrimination on the 312 trial patients

**Apparent (in-sample) performance, not validation:** both coefficient sets were fitted by
Dickson et al. (Hepatology 1989) on these patients (final model: 312 trial + 106 others).

- Outcome: death from any cause; liver transplant (CL) censored. Time zero = registration.
- Harrell's C; Uno's C (IPCW, truncated at τ = 10 y); cumulative/dynamic
  time-dependent AUC at 2, 5, 10 y.
- 95% CI: percentile bootstrap, 1000 patient-level resamples, seed 42.

## Cohort

|                        |   n |
|:-----------------------|----:|
| patients               | 312 |
| deaths                 | 125 |
| transplants (censored) |  19 |
| deaths by 2 y          |  33 |
| at risk at 2 y         | 278 |
| deaths by 5 y          |  85 |
| at risk at 5 y         | 159 |
| deaths by 10 y         | 120 |
| at risk at 10 y        |  32 |

## Discrimination

|                                                               | estimate (95% CI)   |   bootstrap reps used |
|:--------------------------------------------------------------|:--------------------|----------------------:|
| ('Mayo final (Dickson 1989, n=418; standard)', 'harrell_c')   | 0.843 (0.797–0.880) |                  1000 |
| ('Mayo final (Dickson 1989, n=418; standard)', 'uno_c_tau10') | 0.811 (0.770–0.853) |                  1000 |
| ('Mayo final (Dickson 1989, n=418; standard)', 'td_auc_2')    | 0.870 (0.781–0.941) |                  1000 |
| ('Mayo final (Dickson 1989, n=418; standard)', 'td_auc_5')    | 0.917 (0.872–0.955) |                  1000 |
| ('Mayo final (Dickson 1989, n=418; standard)', 'td_auc_10')   | 0.863 (0.796–0.923) |                  1000 |
| ('Mayo initial (Dickson 1989, n=312)', 'harrell_c')           | 0.844 (0.799–0.880) |                  1000 |
| ('Mayo initial (Dickson 1989, n=312)', 'uno_c_tau10')         | 0.810 (0.769–0.852) |                  1000 |
| ('Mayo initial (Dickson 1989, n=312)', 'td_auc_2')            | 0.874 (0.785–0.945) |                  1000 |
| ('Mayo initial (Dickson 1989, n=312)', 'td_auc_5')            | 0.916 (0.871–0.953) |                  1000 |
| ('Mayo initial (Dickson 1989, n=312)', 'td_auc_10')           | 0.858 (0.787–0.920) |                  1000 |
