"""
Feast feature definitions for plant and user embeddings.
21-dim categorical_embedding from feature_engineer (origin_climate, ordinals, water, usda).
"""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32

# Paths relative to feature_repo directory
REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"

plant = Entity(name="plant", join_keys=["plant_id"])
user = Entity(name="user", join_keys=["user_id"])

# 21 feature dims: origin_climate(8) + ordinals(8) + difficulty(1) + water(2) + usda(2)
PLANT_FEATURE_SCHEMA = [
    Field(name="origin_climate_0", dtype=Float32),
    Field(name="origin_climate_1", dtype=Float32),
    Field(name="origin_climate_2", dtype=Float32),
    Field(name="origin_climate_3", dtype=Float32),
    Field(name="origin_climate_4", dtype=Float32),
    Field(name="origin_climate_5", dtype=Float32),
    Field(name="origin_climate_6", dtype=Float32),
    Field(name="origin_climate_7", dtype=Float32),
    Field(name="ideal_light", dtype=Float32),
    Field(name="tolerated_light", dtype=Float32),
    Field(name="ideal_soil", dtype=Float32),
    Field(name="tolerated_soil", dtype=Float32),
    Field(name="size", dtype=Float32),
    Field(name="growth", dtype=Float32),
    Field(name="temperature", dtype=Float32),
    Field(name="care_level", dtype=Float32),
    Field(name="difficulty_score", dtype=Float32),
    Field(name="ideal_water_norm", dtype=Float32),
    Field(name="tolerated_water_norm", dtype=Float32),
    Field(name="usda_min_norm", dtype=Float32),
    Field(name="usda_max_norm", dtype=Float32),
]

plant_features_source = FileSource(
    name="plant_features_source",
    path=str(DATA_DIR / "plant_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

user_features_source = FileSource(
    name="user_features_source",
    path=str(DATA_DIR / "user_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

plant_features_fv = FeatureView(
    name="plant_features",
    entities=[plant],
    ttl=timedelta(days=365),
    schema=PLANT_FEATURE_SCHEMA,
    online=True,
    source=plant_features_source,
)

user_features_fv = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=365),
    schema=PLANT_FEATURE_SCHEMA,  # same 21 dims
    online=True,
    source=user_features_source,
)

# Learned plant tower vectors (64-d) from TwoTowerModel.encode_plant — written after training.
PLANT_TOWER_DIM = 64
PLANT_TOWER_SCHEMA = [
    Field(name=f"tower_{i}", dtype=Float32) for i in range(PLANT_TOWER_DIM)
]

plant_tower_source = FileSource(
    name="plant_tower_source",
    path=str(DATA_DIR / "plant_tower_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

plant_tower_features_fv = FeatureView(
    name="plant_tower_features",
    entities=[plant],
    ttl=timedelta(days=365),
    schema=PLANT_TOWER_SCHEMA,
    online=True,
    source=plant_tower_source,
)
