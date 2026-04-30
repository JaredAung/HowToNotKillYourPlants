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

# Must match feature_repo/feature_definitions.py PLANT_TOWER_DIM and TwoTowerModel OUTPUT_DIM.
PLANT_TOWER_DIM = 64
PLANT_TOWER_FEATURE_NAMES = [f"tower_{i}" for i in range(PLANT_TOWER_DIM)]


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


def plant_tower_embeddings_from_mongo_plants(plants: list[dict]) -> dict[int, list[float]]:
    """
    Build plant_id -> 64-d list from Mongo plant docs (field ``plant_tower_embedding``).
    Skips rows with missing or wrong-length vectors.
    """
    out: dict[int, list[float]] = {}
    for p in plants:
        emb = p.get("plant_tower_embedding")
        if not emb or len(emb) != PLANT_TOWER_DIM:
            continue
        pid = p.get("plant_id")
        if pid is None:
            pid = p.get("id")
        if pid is None:
            continue
        out[int(pid)] = [float(emb[i]) for i in range(PLANT_TOWER_DIM)]
    return out


def push_features_to_feast(
    plants: list[dict],
    users: list[dict],
    repo_path: str | Path | None = None,
    plant_tower_by_id: dict[int, list[float]] | None = None,
) -> None:
    """
    Write plant and user features to parquet and materialize into Feast online store.

    If ``plant_tower_by_id`` is set (non-None), also writes ``plant_tower_features.parquet``
    (64-d tower_* columns). Pass ``None`` to leave that file unchanged from a prior run.
    """
    if repo_path is None:
        repo_path = Path(__file__).resolve().parent.parent.parent / "feature_repo"
    repo_path = Path(repo_path)
    data_dir = repo_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    plant_df = _to_plant_dataframe(plants)
    user_df = _to_user_dataframe(users)

    if plant_df.empty and user_df.empty and plant_tower_by_id is None:
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

    if plant_tower_by_id is not None:
        tower_path = data_dir / "plant_tower_features.parquet"
        df_tower = _plant_tower_to_dataframe(plant_tower_by_id)
        if df_tower.empty:
            _write_empty_plant_tower_parquet(tower_path)
        else:
            df_tower.to_parquet(tower_path, index=False)

    # Run feast apply and materialize
    try:
        subprocess.run(["feast", "apply"], cwd=repo_path, check=True, capture_output=True)
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        subprocess.run(["feast", "materialize-incremental", now], cwd=repo_path, check=True, capture_output=True)
    except FileNotFoundError:
        raise ImportError("feast CLI not found. Run: pip install feast")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Feast command failed: {e.stderr.decode() if e.stderr else e}")


def _plant_tower_to_dataframe(emb_by_id: dict[int, list[float]]) -> pd.DataFrame:
    """Build DataFrame for plant_tower_features (64-d per plant)."""
    now = datetime.utcnow()
    rows = []
    for plant_id, emb in emb_by_id.items():
        if not emb or len(emb) != PLANT_TOWER_DIM:
            continue
        row = {"plant_id": int(plant_id), "event_timestamp": now, "created": now}
        for i, name in enumerate(PLANT_TOWER_FEATURE_NAMES):
            row[name] = float(emb[i])
        rows.append(row)
    return pd.DataFrame(rows)


def _write_empty_plant_tower_parquet(path: Path) -> None:
    cols = ["plant_id", "event_timestamp", "created"] + PLANT_TOWER_FEATURE_NAMES
    pd.DataFrame(columns=cols).to_parquet(path, index=False)


def push_plant_tower_embeddings_to_feast(
    emb_by_id: dict[int, list[float]],
    repo_path: str | Path | None = None,
) -> None:
    """
    Write model-computed plant tower embeddings (64-d) to plant_tower_features.parquet
    and run feast apply + materialize-incremental.
    """
    if repo_path is None:
        repo_path = Path(__file__).resolve().parent.parent.parent / "feature_repo"
    repo_path = Path(repo_path)
    data_dir = repo_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tower_path = data_dir / "plant_tower_features.parquet"

    df = _plant_tower_to_dataframe(emb_by_id)
    if df.empty:
        _write_empty_plant_tower_parquet(tower_path)
    else:
        df.to_parquet(tower_path, index=False)

    try:
        subprocess.run(["feast", "apply"], cwd=str(repo_path), check=True, capture_output=True)
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        subprocess.run(
            ["feast", "materialize-incremental", now],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise ImportError("feast CLI not found. Run: pip install feast")
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Feast command failed: {err}")
