"""Track A end to end: load PBC -> EDA -> Mayo risk score evaluation -> run manifest.

Usage: ``uv run python -m cirrhosis_ml.pbc.run_all [--n-boot 1000] [--refresh]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from cirrhosis_ml.config import (
    N_BOOTSTRAP,
    PBC_CACHE_DIR,
    PBC_REPORTS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from cirrhosis_ml.evaluation.survival_metrics import SurvivalData, bootstrap_metrics
from cirrhosis_ml.pbc.eda import run_eda
from cirrhosis_ml.pbc.load import CACHE_FILENAME, load_pbc
from cirrhosis_ml.pbc.mayo_score import (
    MAYO_COEFFICIENTS,
    MAYO_INITIAL_COEFFICIENTS,
    mayo_score_from_pbc,
)

logger = logging.getLogger(__name__)

HORIZONS_YEARS = (2.0, 5.0, 10.0)
UNO_TAU_YEARS = 10.0
PACKAGES = ("numpy", "pandas", "scikit-learn", "scikit-survival", "lifelines", "matplotlib",
            "ucimlrepo")
SCORE_VARIANTS = {
    "Mayo final (Dickson 1989, n=418; standard)": MAYO_COEFFICIENTS,
    "Mayo initial (Dickson 1989, n=312)": MAYO_INITIAL_COEFFICIENTS,
}


def _cohort_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = {"patients": len(df), "deaths": int(df["event_death"].sum()),
            "transplants (censored)": int((df["event_cr"] == 2).sum())}
    for h in HORIZONS_YEARS:
        died_by_h = (df["time_years"] <= h) & (df["event_death"] == 1)
        rows[f"deaths by {h:g} y"] = int(died_by_h.sum())
        rows[f"at risk at {h:g} y"] = int((df["time_years"] > h).sum())
    return pd.DataFrame.from_dict(rows, orient="index", columns=["n"])


def evaluate_mayo(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Discrimination of both Mayo coefficient sets on the given patients."""
    results = []
    for label, coefs in SCORE_VARIANTS.items():
        risk = mayo_score_from_pbc(df, coefficients=coefs)
        data = SurvivalData.from_arrays(df["time_years"], df["event_death"], risk)
        table = bootstrap_metrics(data, HORIZONS_YEARS, UNO_TAU_YEARS, n_boot=n_boot, seed=seed)
        results.append(table.assign(score=label).reset_index())
    return pd.concat(results, ignore_index=True).set_index(["score", "metric"])


def _format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estimate (95% CI)": [
                f"{r.estimate:.3f} ({r.ci_lower:.3f}–{r.ci_upper:.3f})"
                for r in metrics.itertuples()
            ],
            "bootstrap reps used": metrics["n_boot_ok"].astype(int).to_numpy(),
        },
        index=metrics.index,
    )


def write_mayo_report(counts: pd.DataFrame, metrics: pd.DataFrame, n_boot: int,
                      out_dir: Path) -> Path:
    path = out_dir / "mayo_score_metrics.md"
    metrics.to_csv(out_dir / "mayo_score_metrics.csv")
    text = f"""# Mayo PBC risk score — discrimination on the 312 trial patients

**Apparent (in-sample) performance, not validation:** both coefficient sets were fitted by
Dickson et al. (Hepatology 1989) on these patients (final model: 312 trial + 106 others).

- Outcome: death from any cause; liver transplant (CL) censored. Time zero = registration.
- Harrell's C; Uno's C (IPCW, truncated at τ = {UNO_TAU_YEARS:g} y); cumulative/dynamic
  time-dependent AUC at {", ".join(f"{h:g}" for h in HORIZONS_YEARS)} y.
- 95% CI: percentile bootstrap, {n_boot} patient-level resamples, seed {RANDOM_SEED}.

## Cohort

{counts.to_markdown()}

## Discrimination

{_format_metrics(metrics).to_markdown()}
"""
    path.write_text(text, encoding="utf-8")
    return path


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _git_dirty() -> bool | None:
    """True if tracked files differ from HEAD (results then do not match ``git_commit``)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())


def _package_versions() -> dict[str, str]:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def write_manifest(
    out_dir: Path, n_boot: int, seed: int, outputs: list[Path], data_file: Path
) -> Path:
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "seed": seed,
        "n_bootstrap": n_boot,
        "data": {"source": "UCI ML Repository id 878", "file": data_file.name,
                 "sha256": hashlib.sha256(data_file.read_bytes()).hexdigest()},
        "outputs": [str(p.relative_to(PROJECT_ROOT)) for p in outputs],
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Track A: PBC load, EDA, Mayo score.")
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--refresh", action="store_true", help="re-download UCI 878")
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be >= 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out_dir = PBC_REPORTS_DIR
    cache_dir = PBC_CACHE_DIR
    df = load_pbc(cache_dir=cache_dir, refresh=args.refresh)
    outputs = run_eda(df, out_dir)

    trial = df[df["is_trial"]]
    metrics = evaluate_mayo(trial, n_boot=args.n_boot, seed=args.seed)
    outputs.append(write_mayo_report(_cohort_counts(trial), metrics, args.n_boot, out_dir))
    outputs.append(out_dir / "mayo_score_metrics.csv")
    manifest = write_manifest(out_dir, args.n_boot, args.seed, outputs,
                              data_file=cache_dir / CACHE_FILENAME)
    logger.info("Done. Manifest: %s", manifest)
    logger.info("Mayo (final) Harrell's C: %.3f",
                metrics.loc[(next(iter(SCORE_VARIANTS)), "harrell_c"), "estimate"])


if __name__ == "__main__":
    main()
