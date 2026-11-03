"""Track A Phase 2: ML survival models vs the Mayo score on the PBC trial cohort.

Primary analysis: 312 trial patients, death (transplant censored), repeated nested CV
(5 folds x ``n_repeats``). Out-of-fold (OOF) 5-year risks are averaged over repeats and used for
bootstrap-CI discrimination, calibration and decision curves. Final models are refitted on all
312 patients only for explanation (SHAP, importance) and assumption checks.
Sensitivity: all 418 patients (imputation inside folds), composite endpoint (death or
transplant), and transplant as a competing event (Aalen-Johansen).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.util import Surv

from cirrhosis_ml.evaluation.calibration import CalibrationSummary, calibrate, calibration_slope
from cirrhosis_ml.evaluation.dca import decision_curves
from cirrhosis_ml.evaluation.sample_size import (
    max_r2_cs_survival,
    r2_cs_from_lr,
    survival_sample_size,
)
from cirrhosis_ml.evaluation.survival_metrics import (
    SurvivalData,
    bootstrap_difference,
    bootstrap_metrics,
    point_metrics,
)
from cirrhosis_ml.pbc.mayo_score import MAYO_INITIAL_COEFFICIENTS
from cirrhosis_ml.pbc.models import FULL_FEATURES, MAYO_FEATURES, model_specs
from cirrhosis_ml.reporting import survival_plots as plots
from cirrhosis_ml.survival.competing import cif_curves, grouped_cif_calibration
from cirrhosis_ml.survival.cv import CVConfig, CVResult, fit_spec, run_nested_cv, summarise_folds
from cirrhosis_ml.survival.explain import (
    permutation_importance_table,
    ph_test,
    shap_values,
)

logger = logging.getLogger(__name__)

DICKSON_INITIAL_LR_CHI2 = 199.2  # Dickson 1989 p4, initial 5-variable model, n = 312
DCA_REPORT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4)
SHAP_MODELS = ("gbsa", "rsf")
SHAP_DEPENDENCE = ("Bilirubin", "Albumin", "Prothrombin", "age_years")
IMPORTANCE_MODELS = ("cox_full", "coxnet", "rsf", "gbsa")
REFERENCE_MODEL = "mayo_refit"  # fair benchmark for paired comparisons


@dataclass(frozen=True)
class Phase2Config:
    n_repeats: int = 10
    sens_repeats: int = 5
    n_boot: int = 1000
    seed: int = 42
    horizon: float = 5.0
    shap: bool = True


@dataclass
class Phase2Results:
    labels: dict[str, str]
    sample_size: pd.DataFrame
    fold_summary: pd.DataFrame
    discrimination: pd.DataFrame
    differences: pd.DataFrame
    calibration: pd.DataFrame
    dca: pd.DataFrame
    ph_tests: dict[str, pd.DataFrame]
    mayo_coefficients: pd.DataFrame
    importance: pd.DataFrame
    sensitivity: pd.DataFrame
    competing: pd.DataFrame
    figures: dict[str, Path] = field(default_factory=dict)


def _cv_config(cfg: Phase2Config, n_repeats: int) -> CVConfig:
    return CVConfig(horizon=cfg.horizon, n_repeats=n_repeats, seed=cfg.seed)


def sample_size_table(trial: pd.DataFrame, horizon: float) -> pd.DataFrame:
    rate = trial["event_death"].sum() / trial["time_years"].sum()
    fup = trial["time_years"].mean()
    scenarios = {"Mayo LR chi2 (optimistic)": r2_cs_from_lr(DICKSON_INITIAL_LR_CHI2, 312),
                 "15% of max R2_CS (conservative)": 0.15 * max_r2_cs_survival(rate, fup)}
    rows = []
    for name, r2 in scenarios.items():
        for p in (len(MAYO_FEATURES.columns), len(FULL_FEATURES.columns)):
            res = survival_sample_size(r2, p, rate, horizon, fup)
            rows.append({"scenario": name, "R2_CS": round(r2, 3), "parameters": p,
                         "n_required": res.n_required,
                         "events_required": round(res.events_required),
                         "EPP": round(res.events_per_parameter, 1), "available_n": len(trial),
                         "available_events": int(trial["event_death"].sum())})
    return pd.DataFrame(rows)


def evaluate_oof(time: np.ndarray, event: np.ndarray, oof: pd.DataFrame, cfg: Phase2Config,
                 cv_cfg: CVConfig) -> tuple[pd.DataFrame, dict[str, CalibrationSummary]]:
    disc, cals = [], {}
    for name in oof.columns:
        data = SurvivalData.from_arrays(time, event, oof[name])
        disc.append(bootstrap_metrics(data, cv_cfg.auc_times, cv_cfg.tau, n_boot=cfg.n_boot,
                                      seed=cfg.seed).assign(model=name).reset_index())
        cals[name] = calibrate(time, event, oof[name], cfg.horizon)
    return pd.concat(disc).set_index(["model", "metric"]), cals


def _calibration_table(cals: dict[str, CalibrationSummary]) -> pd.DataFrame:
    return pd.DataFrame({name: {"O/E": c.oe_ratio, "slope": c.slope, "slope_lo": c.slope_ci[0],
                                "slope_hi": c.slope_ci[1]} for name, c in cals.items()}).T


def per_repeat_slopes(cv: CVResult, time: np.ndarray, event: np.ndarray,
                      horizon: float) -> pd.Series:
    """Mean calibration slope over repeats (each repeat's OOF risks, not averaged)."""
    slopes = {col: calibration_slope(time, event, cv.oof_risk[col], horizon)[0]
              for col in cv.oof_risk.columns}
    return pd.Series(slopes).groupby(level=0, sort=False).mean()


def paired_differences(time: np.ndarray, event: np.ndarray, oof: pd.DataFrame,
                       cfg: Phase2Config, cv_cfg: CVConfig) -> pd.DataFrame:
    """Each model minus the refitted Mayo-variable Cox, paired bootstrap CI."""
    ref = oof[REFERENCE_MODEL]
    return pd.concat({
        m: bootstrap_difference(SurvivalData.from_arrays(time, event, oof[m]), ref,
                                cv_cfg.auc_times, cv_cfg.tau, cfg.n_boot, cfg.seed)
        for m in oof.columns if m != REFERENCE_MODEL}, names=["model"])


def mayo_coefficient_table(pipeline) -> pd.DataFrame:
    """Refitted Mayo-variable Cox coefficients (per original unit) vs Dickson initial model."""
    prep = pipeline.named_steps["prep"]
    names = prep.named_steps["select"].get_feature_names_out()
    coef = pipeline.named_steps["model"].coef_ / prep.named_steps["scale"].scale_
    dickson = {"log_Bilirubin": "log_bilirubin", "log_Albumin": "log_albumin", "age_years": "age",
               "log_Prothrombin": "log_prothrombin", "edema_score": "edema"}
    return pd.DataFrame({"refit_312": coef,
                         "dickson_initial_312": [MAYO_INITIAL_COEFFICIENTS[dickson[n]]
                                                 for n in names]}, index=names)


def _fit_final(trial: pd.DataFrame, cfg: Phase2Config) -> dict:
    y = Surv.from_arrays(trial["event_death"].astype(bool), trial["time_years"])
    cv_cfg = _cv_config(cfg, 1)
    return {s.name: fit_spec(s, trial, y, cv_cfg, cfg.seed)[0] for s in model_specs()}


def explain(trial: pd.DataFrame, finals: dict, cfg: Phase2Config, labels: dict[str, str],
            fig_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    y = Surv.from_arrays(trial["event_death"].astype(bool), trial["time_years"])
    cols = FULL_FEATURES.columns
    importance = pd.concat(
        {labels[m]: permutation_importance_table(finals[m], trial, y, cols, cfg.seed)
         ["importance_mean"] for m in IMPORTANCE_MODELS}, axis=1)
    figures: dict[str, Path] = {}
    if cfg.shap:
        for m in SHAP_MODELS:
            sv = shap_values(finals[m], trial, cols, cfg.seed)
            for i, p in enumerate(plots.shap_figures(sv, labels[m], SHAP_DEPENDENCE,
                                                     fig_dir / f"shap_{m}")):
                figures[f"shap_{m}_{i}"] = p
    return importance.round(4), figures


def sensitivity_run(df: pd.DataFrame, event_col: str, label: str, cfg: Phase2Config
                    ) -> pd.DataFrame:
    cv_cfg = _cv_config(cfg, cfg.sens_repeats)
    event = (df[event_col] > 0).to_numpy()
    res = run_nested_cv(df, df["time_years"], event, model_specs(), cv_cfg)
    rows = []
    for name, risk in res.mean_oof_risk().items():
        data = SurvivalData.from_arrays(df["time_years"], event, risk)
        m = point_metrics(data, (cfg.horizon,), cv_cfg.tau)
        cal = calibrate(df["time_years"], event, risk, cfg.horizon)
        rows.append({"analysis": label, "model": name, "n": len(df), "events": int(event.sum()),
                     "harrell_c": m["harrell_c"], "uno_c": m[f"uno_c_tau{cv_cfg.tau:g}"],
                     f"td_auc_{cfg.horizon:g}": m[f"td_auc_{cfg.horizon:g}"],
                     "O/E": cal.oe_ratio, "slope": cal.slope})
    return pd.DataFrame(rows)


def competing_risk(trial: pd.DataFrame, oof: pd.DataFrame, cfg: Phase2Config,
                   fig_dir: Path) -> tuple[pd.DataFrame, Path]:
    t, codes = trial["time_years"].to_numpy(), trial["event_cr"].to_numpy()
    rows = []
    for name in ("mayo_refit", "cox_full", "gbsa"):
        aj = grouped_cif_calibration(t, codes, oof[name], cfg.horizon)
        km = calibrate(t, (codes == 1).astype(int), oof[name], cfg.horizon).table
        rows.append(aj.assign(model=name, observed_km=km["observed_km"].to_numpy()))
    curves = cif_curves(t, codes, {1: "Death", 2: "Transplant"})
    kmf = KaplanMeierFitter().fit(t, codes == 1)  # transplant treated as censoring
    km = 1 - kmf.survival_function_at_times(curves.index.to_numpy())
    km.index = curves.index
    path = plots.cif_plot(curves, km, fig_dir / "competing_risk_cif.png")
    return pd.concat(rows, ignore_index=True), path


def run_phase2(df: pd.DataFrame, out_dir: Path, cfg: Phase2Config) -> tuple[Phase2Results,
                                                                          CVResult]:
    fig_dir = out_dir / "figures"
    labels = {s.name: s.label for s in model_specs()}
    trial = df[df["is_trial"]].reset_index(drop=True)
    time, event = trial["time_years"].to_numpy(), trial["event_death"].to_numpy()
    cv_cfg = _cv_config(cfg, cfg.n_repeats)

    logger.info("Primary nested CV: %d repeats", cfg.n_repeats)
    cv = run_nested_cv(trial, time, event, model_specs(), cv_cfg)
    oof = cv.mean_oof_risk()
    disc, cals = evaluate_oof(time, event, oof, cfg, cv_cfg)
    dca = decision_curves(time, event, {n: oof[n] for n in oof.columns}, cfg.horizon)
    figures = {
        "calibration": plots.calibration_grid(cals, labels, cfg.horizon,
                                              fig_dir / "calibration_5y.png"),
        "dca": plots.decision_curve_plot(dca, labels, cfg.horizon, fig_dir / "dca_5y.png"),
    }

    logger.info("Final models, assumption checks and explanations")
    finals = _fit_final(trial, cfg)
    ph = {m: ph_test(finals[m], trial, time, event) for m in ("mayo_refit", "cox_full")}
    importance, shap_figs = explain(trial, finals, cfg, labels, fig_dir)
    figures.update(shap_figs)

    logger.info("Sensitivity analyses: %d repeats each", cfg.sens_repeats)
    sens = pd.concat([
        sensitivity_run(df.reset_index(drop=True), "event_death",
                        "All 418 (imputation in folds)", cfg),
        sensitivity_run(trial, "event_cr", "Composite: death or transplant (312)", cfg),
    ], ignore_index=True)
    competing, cif_path = competing_risk(trial, oof, cfg, fig_dir)
    figures["cif"] = cif_path

    results = Phase2Results(
        labels=labels, sample_size=sample_size_table(trial, cfg.horizon),
        fold_summary=summarise_folds(cv.fold_metrics), discrimination=disc,
        calibration=_calibration_table(cals).assign(
            slope_per_repeat=per_repeat_slopes(cv, time, event, cfg.horizon)),
        differences=paired_differences(time, event, oof, cfg, cv_cfg),
        dca=dca.loc[list(DCA_REPORT_THRESHOLDS)],
        ph_tests=ph, mayo_coefficients=mayo_coefficient_table(finals["mayo_refit"]),
        importance=importance, sensitivity=sens, competing=competing, figures=figures)
    return results, cv
