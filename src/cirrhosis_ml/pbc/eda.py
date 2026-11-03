"""Exploratory data analysis for the PBC dataset: table 1, missingness, Kaplan-Meier curves.

All outputs are aggregate and written under ``reports/pbc/``. Markdown tables are committed;
CSV copies are git-ignored (``reports/**/*.csv``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.plotting import add_at_risk_counts  # noqa: E402
from lifelines.statistics import multivariate_logrank_test  # noqa: E402

from cirrhosis_ml.config import PBC_REPORTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)

STATUS_LABELS = {0: "Censored (C)", 1: "Died (D)", 2: "Transplant (CL)"}
CONTINUOUS = (
    "age_years", "time_years", "Bilirubin", "Albumin", "Prothrombin", "Cholesterol", "Copper",
    "Alk_Phos", "SGOT", "Tryglicerides", "Platelets",
)
BINARY = ("sex_male", "drug_dpen", "ascites", "hepatomegaly", "spiders")
CATEGORICAL = ("edema_score", "Stage")
KM_HORIZON_YEARS = 12
FIG_DPI = 150


def _fmt_continuous(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "—"
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.2f} [{q1:.2f}–{q3:.2f}]"


def _fmt_count(mask: pd.Series, denom: int) -> str:
    n = int(mask.sum())
    return f"{n} ({100 * n / denom:.1f}%)" if denom else "—"


def table_one(df: pd.DataFrame, group_col: str = "event_cr") -> pd.DataFrame:
    """Table 1 by group: median [IQR] for continuous, n (%) for binary/categorical."""
    groups = {"Overall": df} | {
        STATUS_LABELS.get(k, str(k)): g for k, g in df.groupby(group_col, sort=True)
    }
    rows: dict[str, dict[str, str]] = {"n": {name: str(len(g)) for name, g in groups.items()}}
    for col in CONTINUOUS:
        rows[f"{col}, median [IQR]"] = {name: _fmt_continuous(g[col]) for name, g in groups.items()}
    for col in BINARY:
        rows[f"{col} = 1, n (%)"] = {
            name: _fmt_count(g[col] == 1, int(g[col].notna().sum())) for name, g in groups.items()
        }
    for col in CATEGORICAL:
        for level in sorted(df[col].dropna().unique()):
            rows[f"{col} = {level:g}, n (%)"] = {
                name: _fmt_count(g[col] == level, int(g[col].notna().sum()))
                for name, g in groups.items()
            }
    return pd.DataFrame.from_dict(rows, orient="index")


def missingness(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Missing n (%) overall and by trial participation."""
    parts = {"Overall": df, "Trial (n=312)": df[df["is_trial"]], "Non-trial": df[~df["is_trial"]]}
    return pd.DataFrame(
        {
            name: [_fmt_count(part[c].isna(), len(part)) for c in columns]
            for name, part in parts.items()
        },
        index=list(columns),
    ).rename_axis("variable")


def _km_figure(df: pd.DataFrame, by: str | None, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fitters = []
    groups = [("All patients", df)] if by is None else [
        (f"{by} {k:g}", g) for k, g in df.dropna(subset=[by]).groupby(by)
    ]
    for label, g in groups:
        kmf = KaplanMeierFitter(label=f"{label} (n={len(g)})")
        kmf.fit(g["time_years"], g["event_death"])
        kmf.plot_survival_function(ax=ax, ci_show=by is None)
        fitters.append(kmf)
    if by is not None:
        sub = df.dropna(subset=[by])
        p = multivariate_logrank_test(sub["time_years"], sub[by], sub["event_death"]).p_value
        title = f"{title} (log-rank p = {p:.2g})"
    ax.set(title=title, xlabel="Years since registration", ylabel="Survival probability",
           xlim=(0, KM_HORIZON_YEARS), ylim=(0, 1.02))
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"])
    fig.tight_layout()
    return fig


def _write_table(table: pd.DataFrame, path_stem: Path, heading: str, note: str) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path_stem.with_suffix(".csv"))
    path_stem.with_suffix(".md").write_text(
        f"# {heading}\n\n{note}\n\n{table.to_markdown()}\n", encoding="utf-8"
    )


def run_eda(df: pd.DataFrame, out_dir: Path = PBC_REPORTS_DIR) -> list[Path]:
    """Write table 1, missingness and KM figures; return the paths written."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    note = ("Source: UCI 878 (Mayo PBC, CC BY 4.0). Death = event; transplant (CL) censored "
            "in the primary analysis.")

    _write_table(table_one(df), out_dir / "table1_by_status_all418",
                 "Table 1 — all 418 patients by status", note)
    _write_table(table_one(df[df["is_trial"]]), out_dir / "table1_by_status_trial312",
                 "Table 1 — 312 trial patients by status", note)
    _write_table(missingness(df, CONTINUOUS + BINARY + CATEGORICAL), out_dir / "missingness",
                 "Missingness by trial participation", note)

    written = [out_dir / f"{s}.md" for s in
               ("table1_by_status_all418", "table1_by_status_trial312", "missingness")]
    for by, name, title in ((None, "km_overall", "Overall survival, 418 patients"),
                            ("Stage", "km_by_stage", "Survival by histologic stage")):
        fig = _km_figure(df, by, title)
        path = fig_dir / f"{name}.png"
        fig.savefig(path, dpi=FIG_DPI)
        plt.close(fig)
        written.append(path)
    logger.info("EDA written to %s", out_dir)
    return written
