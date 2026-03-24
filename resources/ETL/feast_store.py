"""
Push plant and user features to Feast feature store.
Writes parquet files and materializes into the online store.
"""
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

FEATURE_NAMES = [
    "origin_climate_0", "origin_climate_1", "origin_climate_2", "origin_climate_3",
    "origin_climate_4", "origin_climate_5", "origin_climate_6", "origin_climate_7",
    "ideal_light", "tolerated_light", "ideal_soil", "tolerated_soil",
    "size", "growth", "temperature", "care_level",
    "difficulty_score", "ideal_water_norm", "tolerated_water_norm",
    "usda_min_norm", "usda_max_norm",
]


def _to_plant_dataframe(plants: list[dict]) -> pd.DataFrame:
    """Convert plants with categorical_embedding to DataFrame for Feast."""
    now = datetime.utcnow()
    rows = []
    for p in plants:
        emb = p.get("categorical_embedding")
        if not emb or len(emb) != 21:
            continue
        plant_id = p.get("id")
        if plant_id is None:
            continue
        row = {"plant_id": int(plant_id), "event_timestamp": now, "created": now}
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(emb):
                row[name] = float(emb[i])
        rows.append(row)
    return pd.DataFrame(rows)


def _write_empty_parquet(path: Path, entity_key: str) -> None:
    """Write empty parquet with correct schema for Feast."""
    cols = [entity_key, "event_timestamp", "created"] + FEATURE_NAMES
    pd.DataFrame(columns=cols).to_parquet(path, index=False)


def _to_user_dataframe(users: list[dict]) -> pd.DataFrame:
    """Convert users with categorical_embedding to DataFrame for Feast."""
    now = datetime.utcnow()
    rows = []
    for u in users:
        emb = u.get("categorical_embedding")
        if not emb or len(emb) != 21:
            continue
        user_id = u.get("user_id")
        if user_id is None:
            continue
        row = {"user_id": int(user_id), "event_timestamp": now, "created": now}
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(emb):
                row[name] = float(emb[i])
        rows.append(row)
    return pd.DataFrame(rows)


def push_features_to_feast(plants: list[dict], users: list[dict], repo_path: str | Path | None = None) -> None:
    """
    Write plant and user features to parquet and materialize into Feast online store.
    """
    if repo_path is None:
        repo_path = Path(__file__).resolve().parent.parent.parent / "feature_repo"
    repo_path = Path(repo_path)
    data_dir = repo_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    plant_df = _to_plant_dataframe(plants)
    user_df = _to_user_dataframe(users)

    if plant_df.empty and user_df.empty:
        return

    # Write parquet (empty schema if no data, so Feast apply/materialize can run)
    plant_path = data_dir / "plant_features.parquet"
    user_path = data_dir / "user_features.parquet"
    if not plant_df.empty:
        plant_df.to_parquet(plant_path, index=False)
    else:
        _write_empty_parquet(plant_path, "plant_id")
    if not user_df.empty:
        user_df.to_parquet(user_path, index=False)
    else:
        _write_empty_parquet(user_path, "user_id")

    # Run feast apply and materialize
    try:
        subprocess.run(["feast", "apply"], cwd=repo_path, check=True, capture_output=True)
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        subprocess.run(["feast", "materialize-incremental", now], cwd=repo_path, check=True, capture_output=True)
    except FileNotFoundError:
        raise ImportError("feast CLI not found. Run: pip install feast")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Feast command failed: {e.stderr.decode() if e.stderr else e}")
