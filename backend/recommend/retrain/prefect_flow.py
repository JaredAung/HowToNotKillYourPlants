"""
Prefect flow wrapper for the retrain pipeline.

Run from project root:
  python -m backend.recommend.retrain.prefect_flow
  python -m backend.recommend.retrain.prefect_flow --dvc-push
  python -m backend.recommend.retrain.prefect_flow --no-use-eval

Or schedule with:
  prefect deploy backend/recommend/retrain/prefect_flow.py:retrain_flow --cron "0 2 * * *"  # daily at 2am
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, task

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = ROOT / "resources" / "two_tower_training" / "output"
BASELINE_VS_PATH = OUTPUT_DIR / "baselineVs.txt"


@task
def run_retrain(include_real: bool = True, update_mongo: bool = True, dvc_add: bool = True) -> bool:
    """Run the retrain script. Returns True on success."""
    args = ["python", "-m", "backend.recommend.retrain.retrain_two_tower"]
    if not include_real:
        args.append("--no-include-real")
    if not update_mongo:
        args.append("--no-update-mongo")
    if dvc_add:
        args.append("--dvc-add")

    result = subprocess.run(args, cwd=str(ROOT))
    return result.returncode == 0


@task
def run_eval(output_path: Path) -> bool:
    """Run baseline vs rec pipeline eval and write results to file."""
    result = subprocess.run(
        ["python", "-m", "backend.eval.eval", "--output", str(output_path)],
        cwd=str(ROOT),
    )
    return result.returncode == 0


@task
def run_dvc_add(path: Path) -> bool:
    """Add a file to DVC."""
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "add", str(path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"dvc add failed (exit {result.returncode}):")
        if result.stderr:
            print(result.stderr)
        return False
    return True


@task
def run_dvc_push() -> bool:
    """Push DVC-tracked files to remote (e.g. Google Drive)."""
    # Use same Python as flow to ensure dvc is available
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "push"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"dvc push failed (exit {result.returncode}):")
        if result.stderr:
            print(result.stderr)
        if result.stdout:
            print(result.stdout)
        return False
    return True


@flow(name="retrain-two-tower", log_prints=True)
def retrain_flow(
    include_real: bool = True,
    update_mongo: bool = False,
    dvc_add: bool = True,
    dvc_push: bool = True,
    use_eval: bool = True,
) -> dict:
    """
    Retrain the two-tower model, run baseline vs rec eval, and optionally push to DVC remote.

    Flow: retrain -> [eval (baselineVs.txt)] -> dvc add [baselineVs] -> dvc push (if enabled)

    Args:
        include_real: Use real garden/death data from MongoDB.
        update_mongo: Update PlantCollection with new embeddings.
        dvc_add: Add model, metrics, and baselineVs.txt to DVC.
        dvc_push: Push to DVC remote after retrain (requires dvc_add=True).
        use_eval: Run baseline vs rec pipeline eval and add baselineVs.txt to DVC.
    """
    success = run_retrain(
        include_real=include_real,
        update_mongo=update_mongo,
        dvc_add=dvc_add,
    )
    if not success:
        return {"status": "failed", "step": "retrain"}

    if use_eval:
        eval_ok = run_eval(BASELINE_VS_PATH)
        if not eval_ok:
            return {"status": "failed", "step": "eval"}

    if dvc_add:
        if use_eval:
            add_ok = run_dvc_add(BASELINE_VS_PATH)
            if not add_ok:
                return {"status": "failed", "step": "dvc_add"}
        if dvc_push:
            push_ok = run_dvc_push()
            if not push_ok:
                return {"status": "failed", "step": "dvc_push"}

    result = {
        "status": "success",
        "model_path": str(OUTPUT_DIR / "two_tower.pt"),
        "metrics_path": str(OUTPUT_DIR / "retrain_metrics.txt"),
    }
    if use_eval:
        result["baseline_vs_path"] = str(BASELINE_VS_PATH)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-include-real", action="store_true", help="Synthetic data only")
    parser.add_argument("--update-mongo", action="store_true", help="Update MongoDB PlantCollection")
    parser.add_argument("--no-dvc-add", action="store_true", help="Skip DVC add")
    parser.add_argument("--dvc-push", action="store_true", help="Push to DVC remote after retrain")
    parser.add_argument("--no-use-eval", action="store_true", help="Skip baseline vs rec eval")
    args = parser.parse_args()
    retrain_flow(
        include_real=not args.no_include_real,
        update_mongo=args.update_mongo,
        dvc_add=not args.no_dvc_add,
        dvc_push=args.dvc_push,
        use_eval=not args.no_use_eval,
    )
