"""
User feature normalization and two-tower inference.
Same vocabs and encoding as resources/two_tower_training/two_tower_training.py.
Converts MongoDB user profile -> user embedding for scoring against plant_tower_embedding.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabs (must match training exactly)
LIGHT_VOCAB = ["direct", "bright_light", "bright_indirect", "indirect", "diffused"]
HUMIDITY_VOCAB = ["low", "medium", "high"]
CARE_LEVEL_VOCAB = ["easy", "medium", "hard"]
SIZE_VOCAB = ["small", "medium", "large"]
CLIMATE_VOCAB = ["Arid Tropical", "Subtropical", "Subtropical arid", "Tropical", "Tropical humid"]
WATER_VOCAB = ["low", "medium", "high"]

# Normalize common variants from frontend/API to canonical vocab
VALUE_NORM: dict[str, dict[str, str]] = {
    "light": {
        "bright light": "bright_light", "bright_light": "bright_light",
        "bright indirect": "bright_indirect", "bright_indirect": "bright_indirect",
        "direct": "direct", "indirect": "indirect", "diffused": "diffused",
    },
    "humidity": {"low": "low", "medium": "medium", "high": "high", "moderate": "medium"},
    "care": {"easy": "easy", "medium": "medium", "hard": "hard", "moderate": "medium"},
    "size": {"small": "small", "medium": "medium", "large": "large"},
    "climate": {
        "arid tropical": "Arid Tropical", "subtropical": "Subtropical",
        "subtropical arid": "Subtropical arid", "tropical": "Tropical",
        "tropical humid": "Tropical humid",
    },
    "water": {"low": "low", "medium": "medium", "high": "high", "moderate": "medium"},
}


def _normalize(val: str | None, key: str) -> str | None:
    if val is None or not str(val).strip():
        return None
    v = str(val).strip()
    v_lower = v.lower().replace(" ", "_")
    norm_map = VALUE_NORM.get(key, {})
    for k, canonical in norm_map.items():
        if k.lower().replace(" ", "_") == v_lower:
            return canonical
    # Pass through if already valid
    return v


def vocab_to_idx(vocab: list[str]) -> dict:
    return {v: i + 1 for i, v in enumerate(vocab)}  # 0 = padding/unknown


def encode_cat(val: str | None, vocab: dict) -> int:
    if val is None:
        return 0
    return vocab.get(val, 0)


def normalize_profile(profile: dict) -> dict:
    """
    Normalize structured string fields in profile to canonical vocab.
    Returns a new dict with normalized values. In-place for nested dicts.
    """
    import copy
    result = copy.deepcopy(profile)
    env = result.get("environment") or {}
    pref = result.get("preferences") or {}
    care_pref = pref.get("care_preferences") or {}
    constraints = result.get("constraints") or {}

    if env.get("light_level"):
        n = _normalize(env["light_level"], "light")
        if n:
            env["light_level"] = n
    if env.get("humidity_level"):
        n = _normalize(env["humidity_level"], "humidity")
        if n:
            env["humidity_level"] = n
    if pref.get("care_level"):
        n = _normalize(pref["care_level"], "care")
        if n:
            pref["care_level"] = n
    if constraints.get("preferred_size"):
        n = _normalize(constraints["preferred_size"], "size")
        if n:
            constraints["preferred_size"] = n
    if result.get("climate"):
        n = _normalize(result["climate"], "climate")
        if n:
            result["climate"] = n
    if care_pref.get("watering_freq"):
        n = _normalize(care_pref["watering_freq"], "water")
        if n:
            care_pref["watering_freq"] = n
    if care_pref.get("care_freq"):
        n = _normalize(care_pref["care_freq"], "water")
        if n:
            care_pref["care_freq"] = n

    pref["care_preferences"] = care_pref
    result["environment"] = env
    result["preferences"] = pref
    result["constraints"] = constraints
    return result


def temp_features(min_f: float | None, max_f: float | None) -> tuple[float, float]:
    """Return (midpoint, width) normalized. Use 0,0 for missing. Same as training."""
    if min_f is None or max_f is None or max_f <= min_f:
        return 0.0, 0.0
    mid = (min_f + max_f) / 2
    width = max_f - min_f
    mid_norm = (mid - 50) / 40
    width_norm = width / 40
    return mid_norm, width_norm


USER_VOCABS = {
    "light": vocab_to_idx(LIGHT_VOCAB),
    "humidity": vocab_to_idx(HUMIDITY_VOCAB),
    "care_level": vocab_to_idx(CARE_LEVEL_VOCAB),
    "size": vocab_to_idx(SIZE_VOCAB),
    "climate": vocab_to_idx(CLIMATE_VOCAB),
    "watering": vocab_to_idx(WATER_VOCAB),
    "care_freq": vocab_to_idx(WATER_VOCAB),
}


def user_profile_to_features(user: dict) -> tuple[list[int], list[float]]:
    """
    Convert MongoDB user document to (u_cat, u_num) for UserTower.
    user: doc from UserCollection (auth, profile, environment, preferences, constraints, climate)
    Returns: (u_cat as list of 7 ints, u_num as list of 2 floats)
    """
    env = user.get("environment", {}) or {}
    pref = user.get("preferences", {}) or {}
    care_pref = pref.get("care_preferences", {}) or {}
    constraints = user.get("constraints", {}) or {}
    temp = env.get("temperature_pref", {}) or {}

    light = _normalize(env.get("light_level"), "light") or env.get("light_level")
    humidity = _normalize(env.get("humidity_level"), "humidity") or env.get("humidity_level")
    care = _normalize(pref.get("care_level"), "care") or pref.get("care_level")
    size = _normalize(constraints.get("preferred_size"), "size") or constraints.get("preferred_size")
    climate_raw = user.get("climate")
    climate = _normalize(climate_raw, "climate") if climate_raw else climate_raw
    water = _normalize(care_pref.get("watering_freq"), "water") or care_pref.get("watering_freq")
    care_freq = _normalize(care_pref.get("care_freq"), "water") or care_pref.get("care_freq")

    u_cat = [
        encode_cat(light, USER_VOCABS["light"]),
        encode_cat(humidity, USER_VOCABS["humidity"]),
        encode_cat(care, USER_VOCABS["care_level"]),
        encode_cat(size, USER_VOCABS["size"]),
        encode_cat(climate, USER_VOCABS["climate"]),
        encode_cat(water, USER_VOCABS["watering"]),
        encode_cat(care_freq, USER_VOCABS["care_freq"]),
    ]
    min_f = temp.get("min_f")
    max_f = temp.get("max_f")
    if min_f is not None:
        min_f = float(min_f)
    if max_f is not None:
        max_f = float(max_f)
    u_temp_mid, u_temp_width = temp_features(min_f, max_f)
    u_num = [u_temp_mid, u_temp_width]
    return u_cat, u_num


# Model path and constants (must match training)
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "resources" / "two_tower_training" / "output" / "two_tower.pt"
TAU = 0.1  # temperature (must match training)
EMBED_DIM = 8
HIDDEN_DIM = 128
OUTPUT_DIM = 64
DROPOUT = 0.2


class UserTower(nn.Module):
    """Must match training UserTower exactly."""
    def __init__(self):
        super().__init__()
        self.embed_light = nn.Embedding(len(LIGHT_VOCAB) + 1, EMBED_DIM, padding_idx=0)
        self.embed_hum = nn.Embedding(len(HUMIDITY_VOCAB) + 1, EMBED_DIM, padding_idx=0)
        self.embed_care = nn.Embedding(len(CARE_LEVEL_VOCAB) + 1, EMBED_DIM, padding_idx=0)
        self.embed_size = nn.Embedding(len(SIZE_VOCAB) + 1, EMBED_DIM, padding_idx=0)
        self.embed_climate = nn.Embedding(len(CLIMATE_VOCAB) + 1, EMBED_DIM, padding_idx=0)
        self.embed_water = nn.Embedding(len(WATER_VOCAB) + 1, EMBED_DIM, padding_idx=0)
        self.embed_care_freq = nn.Embedding(len(WATER_VOCAB) + 1, EMBED_DIM, padding_idx=0)
        cat_dim = 7 * EMBED_DIM
        in_dim = cat_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, u_cat: torch.Tensor, u_num: torch.Tensor) -> torch.Tensor:
        e_light = self.embed_light(u_cat[:, 0])
        e_hum = self.embed_hum(u_cat[:, 1])
        e_care = self.embed_care(u_cat[:, 2])
        e_size = self.embed_size(u_cat[:, 3])
        e_climate = self.embed_climate(u_cat[:, 4])
        e_water = self.embed_water(u_cat[:, 5])
        e_care_freq = self.embed_care_freq(u_cat[:, 6])
        cat = torch.cat([e_light, e_hum, e_care, e_size, e_climate, e_water, e_care_freq], dim=1)
        x = torch.cat([cat, u_num], dim=1)
        return self.mlp(x)


_user_tower: UserTower | None = None


def get_user_tower() -> UserTower:
    """Lazy-load UserTower from checkpoint."""
    global _user_tower
    if _user_tower is None:
        _user_tower = UserTower()
        if MODEL_PATH.exists():
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            # Extract user_tower weights (keys like "user_tower.embed_light.weight")
            user_state = {k.replace("user_tower.", ""): v for k, v in state.items() if k.startswith("user_tower.")}
            _user_tower.load_state_dict(user_state, strict=True)
        _user_tower.eval()
    return _user_tower


def compute_user_embedding(user: dict) -> list[float]:
    """
    Compute L2-normalized user embedding from MongoDB user doc.
    Returns 64-d list for dot product with plant_tower_embedding.
    """
    u_cat, u_num = user_profile_to_features(user)
    u_cat_t = torch.tensor([u_cat], dtype=torch.long)
    u_num_t = torch.tensor([u_num], dtype=torch.float32)
    with torch.no_grad():
        emb = get_user_tower()(u_cat_t, u_num_t)
        emb = F.normalize(emb, dim=1)
    return emb[0].tolist()


def score_plants(user_embedding: list[float], plant_embeddings: list[tuple[int, list[float]]]) -> list[tuple[int, float]]:
    """
    Score plants by dot product (both embeddings assumed L2-normalized).
    plant_embeddings: [(plant_id, embedding), ...]
    Returns: [(plant_id, score), ...] sorted by score descending.
    """
    u = torch.tensor([user_embedding], dtype=torch.float32)
    scores = []
    for pid, p_emb in plant_embeddings:
        p = torch.tensor([p_emb], dtype=torch.float32)
        s = (u * p).sum().item() / TAU
        scores.append((pid, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
