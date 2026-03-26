"""
Two-tower model for user-plant matching.
Uses the new schema: 21-dim categorical_embedding from feature_engineer.
Separate UserTower and PlantTower with independent weights.
Can pull features from Feast for inference.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Aligned with feature_engineer and Feast feature_definitions
INPUT_DIM = 21  # plant: origin_climate(8) + ordinals(8) + difficulty(1) + water(2) + usda(2)
USER_INPUT_DIM = 21 + 64 + 64  # user: categorical_embedding + plants_grown_agg + plants_killed_agg
FEATURE_NAMES = [
    "origin_climate_0", "origin_climate_1", "origin_climate_2", "origin_climate_3",
    "origin_climate_4", "origin_climate_5", "origin_climate_6", "origin_climate_7",
    "ideal_light", "tolerated_light", "ideal_soil", "tolerated_soil",
    "size", "growth", "temperature", "care_level",
    "difficulty_score", "ideal_water_norm", "tolerated_water_norm",
    "usda_min_norm", "usda_max_norm",
]
DEFAULT_FEAST_REPO = Path(__file__).resolve().parent.parent.parent / "feature_repo"
HIDDEN_DIM = 128
OUTPUT_DIM = 64
# Must match feature_repo/feature_definitions.py plant_tower_features (tower_0..tower_63).
PLANT_TOWER_FEATURE_NAMES = [f"tower_{i}" for i in range(OUTPUT_DIM)]
DROPOUT = 0.2
TAU = 0.1  # temperature for normalized dot product


def _build_tower(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    """Build MLP tower: input -> hidden -> hidden -> output."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class UserTower(nn.Module):
    """User tower. Input: 21-dim categorical_embedding."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.mlp = _build_tower(input_dim, hidden_dim, output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PlantTower(nn.Module):
    """Plant tower. Input: 21-dim categorical_embedding."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.mlp = _build_tower(input_dim, hidden_dim, output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TwoTowerModel(nn.Module):
    """
    Two-tower model for user-plant matching.
    User tower: 149-dim (categorical + plants_grown_agg + plants_killed_agg).
    Plant tower: 21-dim categorical.
    """

    def __init__(
        self,
        user_input_dim: int = USER_INPUT_DIM,
        plant_input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        dropout: float = DROPOUT,
        tau: float = TAU,
    ):
        super().__init__()
        self.tau = tau
        self.user_tower = UserTower(
            input_dim=user_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.plant_tower = PlantTower(
            input_dim=plant_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        user_emb: torch.Tensor,
        plant_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        user_emb: (B, 149) - categorical + plants_grown_agg + plants_killed_agg
        plant_emb: (B, 21) - categorical_embedding from plant
        Returns: (B,) similarity scores (dot product / tau)
        """
        u = self.user_tower(user_emb)
        p = self.plant_tower(plant_emb)
        u = F.normalize(u, dim=1)
        p = F.normalize(p, dim=1)
        return (u * p).sum(dim=1) / self.tau

    def encode_user(self, user_emb: torch.Tensor) -> torch.Tensor:
        """Encode user categorical_embedding to L2-normalized vector."""
        out = self.user_tower(user_emb)
        return F.normalize(out, dim=1)

    def encode_plant(self, plant_emb: torch.Tensor) -> torch.Tensor:
        """Encode plant categorical_embedding to L2-normalized vector."""
        out = self.plant_tower(plant_emb)
        return F.normalize(out, dim=1)

    def encode_user_from_feast(
        self,
        user_id: int,
        fs=None,
        repo_path: str | Path | None = None,
    ) -> torch.Tensor:
        """Fetch user features from Feast and encode to L2-normalized vector."""
        vec = _fetch_features_from_feast("user", user_id, fs, repo_path)
        with torch.no_grad():
            return self.encode_user(vec.unsqueeze(0)).squeeze(0)

    def encode_plant_from_feast(
        self,
        plant_id: int,
        fs=None,
        repo_path: str | Path | None = None,
    ) -> torch.Tensor:
        """Fetch plant features from Feast and encode to L2-normalized vector."""
        vec = _fetch_features_from_feast("plant", plant_id, fs, repo_path)
        with torch.no_grad():
            return self.encode_plant(vec.unsqueeze(0)).squeeze(0)

    def score_from_feast(
        self,
        user_id: int,
        plant_id: int,
        fs=None,
        repo_path: str | Path | None = None,
    ) -> float:
        """Fetch features from Feast and compute similarity score for one user-plant pair."""
        u_vec = _fetch_features_from_feast("user", user_id, fs, repo_path).unsqueeze(0)
        p_vec = _fetch_features_from_feast("plant", plant_id, fs, repo_path).unsqueeze(0)
        with torch.no_grad():
            score = self.forward(u_vec, p_vec).item()
        return score

    def encode_users_from_feast(
        self,
        user_ids: list[int],
        fs=None,
        repo_path: str | Path | None = None,
    ) -> torch.Tensor:
        """Fetch user features from Feast and encode to (N, 64) L2-normalized embeddings."""
        vecs = _fetch_batch_features_from_feast("user", user_ids, fs, repo_path)
        with torch.no_grad():
            return self.encode_user(vecs)

    def encode_plants_from_feast(
        self,
        plant_ids: list[int],
        fs=None,
        repo_path: str | Path | None = None,
    ) -> torch.Tensor:
        """Fetch plant features from Feast and encode to (N, 64) L2-normalized embeddings."""
        vecs = _fetch_batch_features_from_feast("plant", plant_ids, fs, repo_path)
        with torch.no_grad():
            return self.encode_plant(vecs)


def _fetch_batch_features_from_feast(
    entity_type: str,
    entity_ids: list[int],
    fs=None,
    repo_path: str | Path | None = None,
) -> torch.Tensor:
    """Fetch features for multiple entities. Returns (N, 21) tensor."""
    if not entity_ids:
        return torch.empty(0, INPUT_DIM, dtype=torch.float32)
    try:
        from feast import FeatureStore
    except ImportError:
        raise ImportError("feast not installed. Run: pip install feast")

    if fs is None:
        path = repo_path or DEFAULT_FEAST_REPO
        fs = FeatureStore(repo_path=str(path))

    fv_name = "user_features" if entity_type == "user" else "plant_features"
    join_key = "user_id" if entity_type == "user" else "plant_id"
    features = [f"{fv_name}:{name}" for name in FEATURE_NAMES]
    entity_rows = [{join_key: eid} for eid in entity_ids]

    result = fs.get_online_features(
        features=features,
        entity_rows=entity_rows,
    ).to_dict()

    rows = []
    for i in range(len(entity_ids)):
        row = []
        for name in FEATURE_NAMES:
            full_key = f"{fv_name}:{name}"
            key = full_key if full_key in result else name
            if key not in result:
                key = next((k for k in result if k.endswith(name)), None)
            val = result[key][i] if key else 0.0
            row.append(float(val) if val is not None else 0.0)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def fetch_plant_tower_embeddings_from_feast(
    plant_ids: list[int],
    fs=None,
    repo_path: str | Path | None = None,
) -> dict[int, list[float]]:
    """
    Fetch 64-d plant tower vectors from Feast ``plant_tower_features`` (tower_0..tower_63).
    Plants with missing or partial features are omitted from the returned dict.
    """
    if not plant_ids:
        return {}
    try:
        from feast import FeatureStore
    except ImportError:
        raise ImportError("feast not installed. Run: pip install feast")

    if fs is None:
        path = repo_path or DEFAULT_FEAST_REPO
        fs = FeatureStore(repo_path=str(path))

    fv_name = "plant_tower_features"
    join_key = "plant_id"
    features = [f"{fv_name}:{name}" for name in PLANT_TOWER_FEATURE_NAMES]
    entity_rows = [{join_key: eid} for eid in plant_ids]

    result = fs.get_online_features(
        features=features,
        entity_rows=entity_rows,
    ).to_dict()

    out: dict[int, list[float]] = {}
    for i, pid in enumerate(plant_ids):
        row: list[float] = []
        missing = False
        for name in PLANT_TOWER_FEATURE_NAMES:
            full_key = f"{fv_name}:{name}"
            key = full_key if full_key in result else name
            if key not in result:
                key = next((k for k in result if k.endswith(name)), None)
            val = result[key][i] if key else None
            if val is None:
                missing = True
                break
            v = float(val)
            if v != v:  # NaN
                missing = True
                break
            row.append(v)
        if not missing and len(row) == OUTPUT_DIM:
            out[pid] = row
    return out


def _fetch_features_from_feast(
    entity_type: str,
    entity_id: int,
    fs=None,
    repo_path: str | Path | None = None,
) -> torch.Tensor:
    """
    Fetch 21-dim features from Feast for a user or plant.
    entity_type: "user" or "plant"
    entity_id: user_id or plant_id
    Returns: (21,) tensor
    """
    try:
        from feast import FeatureStore
    except ImportError:
        raise ImportError("feast not installed. Run: pip install feast")

    if fs is None:
        path = repo_path or DEFAULT_FEAST_REPO
        fs = FeatureStore(repo_path=str(path))

    fv_name = "user_features" if entity_type == "user" else "plant_features"
    join_key = "user_id" if entity_type == "user" else "plant_id"
    features = [f"{fv_name}:{name}" for name in FEATURE_NAMES]

    result = fs.get_online_features(
        features=features,
        entity_rows=[{join_key: entity_id}],
    ).to_dict()

    # Build 21-dim vector in order (Feast keys: "fv:name" or "name" depending on version)
    vec = []
    for name in FEATURE_NAMES:
        full_key = f"{fv_name}:{name}"
        key = full_key if full_key in result else name
        if key not in result:
            key = next((k for k in result if k.endswith(name)), None)
        if key is None:
            raise KeyError(f"Feature {name} not in Feast response: {list(result.keys())}")
        val = result[key][0]
        vec.append(float(val) if val is not None else 0.0)
    return torch.tensor(vec, dtype=torch.float32)


def load_features_from_feast(
    user_ids: list[int],
    plant_ids: list[int],
    fs=None,
    repo_path: str | Path | None = None,
) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    """
    Fetch 21-dim features from Feast for users and plants.
    Returns (user_by_id, plant_by_id) where each maps id -> categorical_embedding (list of 21 floats).
    """
    user_by_id = {}
    plant_by_id = {}
    if user_ids:
        u_tensor = _fetch_batch_features_from_feast("user", user_ids, fs, repo_path)
        for i, uid in enumerate(user_ids):
            user_by_id[uid] = {"user_id": uid, "categorical_embedding": u_tensor[i].tolist()}
    if plant_ids:
        p_tensor = _fetch_batch_features_from_feast("plant", plant_ids, fs, repo_path)
        for i, pid in enumerate(plant_ids):
            plant_by_id[pid] = {"id": pid, "categorical_embedding": p_tensor[i].tolist()}
    return user_by_id, plant_by_id


def create_two_tower_model(
    user_input_dim: int = USER_INPUT_DIM,
    plant_input_dim: int = INPUT_DIM,
    hidden_dim: int = HIDDEN_DIM,
    output_dim: int = OUTPUT_DIM,
    dropout: float = DROPOUT,
    tau: float = TAU,
) -> TwoTowerModel:
    """Create a TwoTowerModel instance."""
    return TwoTowerModel(
        user_input_dim=user_input_dim,
        plant_input_dim=plant_input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        tau=tau,
    )
