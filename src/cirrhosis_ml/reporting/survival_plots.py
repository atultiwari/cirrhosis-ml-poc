"""Figures for survival-model reports (aggregate outputs only)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from cirrhosis_ml.evaluation.calibration import CalibrationSummary  # noqa: E402

FIG_DPI = 150


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


def calibration_grid(cals: Mapping[str, CalibrationSummary], labels: Mapping[str, str],
                     horizon: float, path: Path) -> Path:
    """One calibration panel per model: grouped predicted vs KM observed risk."""
    n = len(cals)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)
    for ax, (name, cal) in zip(axes.flat, cals.items(), strict=False):
        t = cal.table
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
        ax.errorbar(t["mean_predicted"], t["observed_km"],
                    yerr=[t["observed_km"] - t["observed_lo"], t["observed_hi"] - t["observed_km"]],
                    fmt="o", ms=4, capsize=2)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=labels.get(name, name),
               xlabel=f"Predicted {horizon:g}-y risk", ylabel=f"Observed {horizon:g}-y risk (KM)")
        ax.text(0.03, 0.92, f"O/E {cal.oe_ratio:.2f}\nslope {cal.slope:.2f} "
                f"({cal.slope_ci[0]:.2f}–{cal.slope_ci[1]:.2f})", fontsize=8, va="top",
                transform=ax.transAxes)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return _save(fig, path)


def decision_curve_plot(curves: pd.DataFrame, labels: Mapping[str, str], horizon: float,
                        path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for col in curves.columns:
        style = {"treat_all": dict(c="black", ls=":"), "treat_none": dict(c="grey", ls="--")}
        ax.plot(curves.index, curves[col], label=labels.get(col, col), lw=1.4,
                **style.get(col, {}))
    top = np.nanmax(curves.to_numpy())
    ax.set(ylim=(-0.05, top * 1.1), xlabel=f"Threshold probability ({horizon:g}-y death)",
           ylabel="Net benefit", title=f"Decision curves at {horizon:g} years (out-of-fold)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def cif_plot(curves: pd.DataFrame, km_risk: pd.Series, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for col in curves.columns:
        ax.step(curves.index, curves[col], where="post", label=f"{col} (Aalen-Johansen)")
    ax.step(km_risk.index, km_risk.to_numpy(), where="post", ls="--", c="grey",
            label="Death, 1 − KM (transplant censored)")
    ax.set(xlabel="Years since registration", ylabel="Cumulative incidence", ylim=(0, 1),
           title="Death and transplant as competing events")
    ax.legend(fontsize=8)
    return _save(fig, path)


def shap_figures(explanation: shap.Explanation, title: str, dependence: Sequence[str],
                 stem: Path) -> list[Path]:
    """Beeswarm summary plus dependence plots for selected predictors."""
    paths = []
    shap.plots.beeswarm(explanation, max_display=16, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    paths.append(_save(fig, stem.with_name(stem.name + "_beeswarm.png")))
    names = list(explanation.feature_names)
    fig, axes = plt.subplots(1, len(dependence), figsize=(3.6 * len(dependence), 3.4))
    for ax, feat in zip(np.atleast_1d(axes), dependence, strict=True):
        shap.plots.scatter(explanation[:, names.index(feat)], ax=ax, show=False)
    fig.suptitle(f"{title}: SHAP dependence", fontsize=10)
    paths.append(_save(fig, stem.with_name(stem.name + "_dependence.png")))
    return paths
