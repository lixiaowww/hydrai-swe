from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.predict import evaluate_run_dir


def find_cv_run_dirs(runs_root: str = "src/neuralhydrology/runs") -> List[Path]:
    root = Path(runs_root)
    # Pattern: runs/{experiment_name}_{start_time}/hydrai_swe_cv_fold*_
    patterns = [str(root / "*" / "hydrai_swe_cv_fold*_*"), str(root / "hydrai_swe_cv_fold*_*")]
    found: List[Path] = []
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                pp = Path(p)
                if pp.is_dir():
                    found.append(pp)
            except Exception:
                continue
    # Sort by fold then mtime
    found.sort(key=lambda p: (p.name.split("_fold")[-1][:1], p.stat().st_mtime))
    return found


def collect_metrics() -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for run_dir in find_cv_run_dirs():
        name = run_dir.name
        fold = ""
        if "_fold" in name:
            try:
                fold = name.split("_fold")[1].split("_")[0]
            except Exception:
                fold = ""
        try:
            metrics = evaluate_run_dir(str(run_dir))
            if metrics:
                row: Dict[str, float | str] = {"run_dir": str(run_dir), "fold": fold}
                row.update({k: float(v) for k, v in metrics.items()})
            else:
                row = {"run_dir": str(run_dir), "fold": fold, "status": "pending_or_no_metrics"}
        except Exception as e:
            row = {"run_dir": str(run_dir), "fold": fold, "status": f"error: {e}"}
        rows.append(row)
    return rows


def print_table(rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        print("No CV run directories found.")
        return
    # Determine columns
    cols = ["fold", "run_dir", "NSE", "RMSE", "MAE", "R2", "status"]
    # Print header
    print(",".join(cols))
    for r in rows:
        values = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                values.append(f"{v:.4f}")
            else:
                values.append(str(v))
        print(",".join(values))


if __name__ == "__main__":
    rows = collect_metrics()
    print_table(rows)


