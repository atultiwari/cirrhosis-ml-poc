"""Minimum sample size for developing a survival (time-to-event) prediction model.

Python implementation of the Riley et al. criteria as used by the R/Stata package `pmsampsize`
(``type = "s"``), because the Python port on PyPI requires Python >= 3.12 (see DECISIONS.md).

References:
- Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a multivariable
  prediction model: Part II — binary and time-to-event outcomes. *Stat Med* 2019;38:1276-96.
- Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a
  clinical prediction model. *BMJ* 2020;368:m441 (worked survival example: R2_CS = 0.051,
  30 parameters, rate 0.065/yr, timepoint 2 y, mean follow-up 2.07 y -> n = 5143, 692 events).

Criteria:
1. expected uniform shrinkage >= ``shrinkage`` (default 0.9);
2. small optimism (<= ``delta``, default 0.05) in Nagelkerke R2 — uses max R2_CS implied by the
   overall event rate;
3. precise overall risk at ``timepoint`` — reported only (95% CI of the cumulative incidence
   at the required n), not enforced: ``n_required`` reflects criteria 1-2.

``parameters`` counts candidate-predictor degrees of freedom; flexible learners (forests,
boosting) use more effective parameters, so EPP is optimistic for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.959964
NULL_MODEL_N = 1_000_000  # large n used by pmsampsize to evaluate the null log-likelihood


@dataclass(frozen=True)
class SurvivalSampleSize:
    n_criterion1: int
    n_criterion2: int
    n_required: int
    shrinkage_at_required: float
    events_required: float
    events_per_parameter: float
    max_r2_cs: float
    cuminc_at_timepoint: float
    cuminc_ci: tuple[float, float]


def max_r2_cs_survival(rate: float, mean_followup: float) -> float:
    """Maximum Cox-Snell R2 for a survival model given the event rate and mean follow-up."""
    events = NULL_MODEL_N * rate * mean_followup
    ln_l_null = events * math.log(events / NULL_MODEL_N) - events
    return 1 - math.exp(2 * ln_l_null / NULL_MODEL_N)


def r2_cs_from_lr(lr_chi2: float, n: int) -> float:
    """Cox-Snell R2 from a reported likelihood-ratio chi-square statistic: 1 - exp(-LR/n)."""
    if lr_chi2 <= 0 or n <= 0:
        raise ValueError("lr_chi2 and n must be positive")
    return 1 - math.exp(-lr_chi2 / n)


def _n_for_shrinkage(parameters: int, r2_cs: float, shrinkage: float) -> float:
    return parameters / ((shrinkage - 1) * math.log(1 - r2_cs / shrinkage))


def survival_sample_size(
    r2_cs: float,
    parameters: int,
    rate: float,
    timepoint: float,
    mean_followup: float,
    shrinkage: float = 0.9,
    delta: float = 0.05,
) -> SurvivalSampleSize:
    """Riley et al. minimum sample size for a time-to-event prediction model."""
    if parameters < 1:
        raise ValueError("parameters must be >= 1")
    if not 0 < shrinkage < 1 or not 0 < delta < 1:
        raise ValueError("shrinkage and delta must be in (0, 1)")
    if rate <= 0 or timepoint <= 0 or mean_followup <= 0:
        raise ValueError("rate, timepoint and mean_followup must be > 0")
    max_r2 = max_r2_cs_survival(rate, mean_followup)
    if not 0 < r2_cs < max_r2:
        raise ValueError(f"r2_cs must be in (0, max R2_CS = {max_r2:.3f})")
    if r2_cs >= shrinkage:
        raise ValueError("r2_cs must be below the target shrinkage")

    n1 = math.ceil(_n_for_shrinkage(parameters, r2_cs, shrinkage))
    s_small_optimism = r2_cs / (r2_cs + delta * max_r2)
    n2 = math.ceil(_n_for_shrinkage(parameters, r2_cs, s_small_optimism))
    n_req = max(n1, n2)
    shrink_req = shrinkage if n1 >= n2 else s_small_optimism

    person_time = mean_followup * n_req
    half_width = Z_95 * math.sqrt(rate / person_time)
    events = n_req * rate * mean_followup
    return SurvivalSampleSize(
        n_criterion1=n1,
        n_criterion2=n2,
        n_required=n_req,
        shrinkage_at_required=shrink_req,
        events_required=events,
        events_per_parameter=events / parameters,
        max_r2_cs=max_r2,
        cuminc_at_timepoint=1 - math.exp(-rate * timepoint),
        cuminc_ci=(
            1 - math.exp(-max(rate - half_width, 0.0) * timepoint),
            1 - math.exp(-(rate + half_width) * timepoint),
        ),
    )
