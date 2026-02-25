"""
Two-tower model training for user-plant matching.
User tower: categorical (light, humidity, care_level, preferred_size, climate, watering_freq, care_freq) + numeric (temp_mid, temp_width)
Plant tower: categorical (sunlight_bucket, humidity, water, care_level, size, climate) + numeric (temp_mid, temp_width) + desc_embeddings (Voyage)
"""
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Paths (relative to project root)
ROOT = Path(__file__).resolve().parent.parent.parent
USERS_PATH = ROOT / "resources" / "synthetic_users.json"
PLANTS_PATH = ROOT / "resources" / "plant_profiles.json"
INTERACTIONS_PATH = ROOT / "resources" / "interactions.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Vocab (aligned with schema)
LIGHT_VOCAB = ["direct", "bright_light", "bright_indirect", "indirect", "diffused"]
HUMIDITY_VOCAB = ["low", "medium", "high"]
CARE_LEVEL_VOCAB = ["easy", "medium", "hard"]
SIZE_VOCAB = ["small", "medium", "large"]
CLIMATE_VOCAB = ["Arid Tropical", "Subtropical", "Subtropical arid", "Tropical", "Tropical humid"]
WATER_VOCAB = ["low", "medium", "high"]

DESC_EMBED_DIM = 1024  # Voyage voyage-4-lite
EMBED_DIM = 8  # per categorical
HIDDEN_DIM = 128
OUTPUT_DIM = 64
DROPOUT = 0.2
BATCH_SIZE = 256
EPOCHS = 15
LR = 1e-3
VAL_RATIO = 0.1
TAU = 0.1  # temperature for normalized dot product
EVAL_K = 10
EVAL_KS = [10, 20, 50]
ORACLE_POS_THRESHOLD = 0.75  # plants with oracle score >= this are "true" positives for broader eval
ORACLE_TOP_N_POSITIVES = 50  # alternative: top N by oracle score per user
EARLY_STOP_PATIENCE = 3


# --- Oracle scoring (same logic as interactions.py for label generation) ---
BUCKET_ORDER = ["low", "medium", "high"]
MISSING_NEUTRAL = 0.5


def _bucket_distance(a: str | None, b: str | None) -> int:
    if a not in BUCKET_ORDER or b not in BUCKET_ORDER:
        return 2
    return abs(BUCKET_ORDER.index(a) - BUCKET_ORDER.index(b))


def _light_match(user_light: str | None, plant_ideal: str | None, plant_tolerated: str | None) -> float:
    if not user_light:
        return MISSING_NEUTRAL
    if not plant_ideal and not plant_tolerated:
        return MISSING_NEUTRAL
    if plant_ideal and user_light == plant_ideal:
        return 1.0
    if plant_tolerated and user_light == plant_tolerated:
        return 0.6
    return 0.0


def _water_match(user_watering: str | None, plant_water: str | None) -> float:
    if not user_watering or not plant_water:
        return MISSING_NEUTRAL
    d = _bucket_distance(user_watering, plant_water)
    return 1.0 if d == 0 else (0.6 if d == 1 else 0.0)


def _humidity_match(user_humidity: str | None, plant_humidity: str | None) -> float:
    if not user_humidity or not plant_humidity:
        return MISSING_NEUTRAL
    d = _bucket_distance(user_humidity, plant_humidity)
    return 1.0 if d == 0 else (0.6 if d == 1 else 0.0)


def _temp_overlap(
    user_min: float | None, user_max: float | None,
    plant_min: float | None, plant_max: float | None,
) -> float:
    if user_min is None or user_max is None or plant_min is None or plant_max is None:
        return MISSING_NEUTRAL
    overlap_start = max(user_min, plant_min)
    overlap_end = min(user_max, plant_max)
    overlap_len = max(0.0, overlap_end - overlap_start)
    if overlap_len <= 0:
        return 0.0
    user_range = user_max - user_min
    return min(1.0, overlap_len / user_range) if user_range > 0 else 0.0


def _size_match(user_preferred: str | None, plant_size: str | None) -> float:
    if not user_preferred or not plant_size:
        return MISSING_NEUTRAL
    return 1.0 if user_preferred == plant_size else 0.5


def _hard_filter(user: dict, plant: dict) -> bool:
    hard_no = user.get("constraints", {}).get("hard_no") or []
    if "frequent_watering" in hard_no:
        plant_water = (plant.get("Care", {}) or {}).get("water_req_bucket")
        if plant_water == "high":
            return True
    return False


def compute_oracle_score(user: dict, plant: dict) -> float:
    """Oracle compatibility score 0..1 (same logic as interactions.py)."""
    if _hard_filter(user, plant):
        return 0.0
    env = user.get("environment", {}) or {}
    pref = user.get("preferences", {}) or {}
    care_pref = pref.get("care_preferences", {}) or {}
    constraints = user.get("constraints", {}) or {}
    plant_care = plant.get("Care", {}) or {}
    plant_info = plant.get("Info", {}) or {}
    light_req = plant_care.get("light_req", {}) or {}
    ideal = light_req.get("ideal_light", {}) or {}
    tolerated = light_req.get("tolerated_light", {}) or {}
    lm = _light_match(env.get("light_level"), ideal.get("sunlight_bucket"), tolerated.get("sunlight_bucket"))
    wm = _water_match(care_pref.get("watering_freq"), plant_care.get("water_req_bucket"))
    hm = _humidity_match(env.get("humidity_level"), plant_care.get("humidity_req_bucket"))
    temp_pref = env.get("temperature_pref", {}) or {}
    tm = _temp_overlap(
        temp_pref.get("min_f"), temp_pref.get("max_f"),
        plant_care.get("temp_req", {}).get("min_temp"),
        plant_care.get("temp_req", {}).get("max_temp"),
    )
    sm = _size_match(constraints.get("preferred_size"), plant_info.get("size"))
    return (lm + wm + hm + tm + sm) / 5.0


def vocab_to_idx(vocab: list[str]) -> dict:
    return {v: i + 1 for i, v in enumerate(vocab)}  # 0 = padding/unknown


def encode_cat(val: str | None, vocab: dict) -> int:
    if val is None:
        return 0
    return vocab.get(val, 0)


def temp_features(min_f: float | None, max_f: float | None) -> tuple[float, float]:
    """Return (midpoint, width) normalized. Use 0,0 for missing."""
    if min_f is None or max_f is None or max_f <= min_f:
        return 0.0, 0.0
    mid = (min_f + max_f) / 2
    width = max_f - min_f
    # Normalize: mid typically 50-90, width 5-40
    mid_norm = (mid - 50) / 40  # ~0-1
    width_norm = width / 40  # ~0-1
    return mid_norm, width_norm


class UserPlantDataset(Dataset):
    def __init__(self, interactions: list[dict], users: dict[int, dict], plants: dict[int, dict], user_vocabs: dict, plant_vocabs: dict):
        self.interactions = interactions
        self.users = users
        self.plants = plants
        self.user_vocabs = user_vocabs
        self.plant_vocabs = plant_vocabs

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions[idx]
        uid, pid, label = row["user_id"], row["plant_id"], row["label"]
        user = self.users[uid]
        plant = self.plants[pid]

        # User features
        env = user.get("environment", {}) or {}
        pref = user.get("preferences", {}) or {}
        care_pref = pref.get("care_preferences", {}) or {}
        constraints = user.get("constraints", {}) or {}
        temp = env.get("temperature_pref", {}) or {}

        u_light = encode_cat(env.get("light_level"), self.user_vocabs["light"])
        u_hum = encode_cat(env.get("humidity_level"), self.user_vocabs["humidity"])
        u_care = encode_cat(pref.get("care_level"), self.user_vocabs["care_level"])
        u_size = encode_cat(constraints.get("preferred_size"), self.user_vocabs["size"])
        u_climate = encode_cat(user.get("climate"), self.user_vocabs["climate"])
        u_water = encode_cat(care_pref.get("watering_freq"), self.user_vocabs["watering"])
        u_care_freq = encode_cat(care_pref.get("care_freq"), self.user_vocabs["care_freq"])
        u_temp_mid, u_temp_width = temp_features(temp.get("min_f"), temp.get("max_f"))

        # Plant features
        care = plant.get("Care", {}) or {}
        info = plant.get("Info", {}) or {}
        light_req = care.get("light_req", {}) or {}
        ideal = light_req.get("ideal_light", {}) or {}
        tolerated = light_req.get("tolerated_light", {}) or {}
        temp_req = care.get("temp_req", {}) or {}

        p_light = encode_cat(ideal.get("sunlight_bucket"), self.plant_vocabs["light"])
        p_tolerated = encode_cat(tolerated.get("sunlight_bucket"), self.plant_vocabs["light"])
        p_hum = encode_cat(care.get("humidity_req_bucket"), self.plant_vocabs["humidity"])
        p_water = encode_cat(care.get("water_req_bucket"), self.plant_vocabs["water"])
        p_care = encode_cat(care.get("care_level"), self.plant_vocabs["care_level"])
        p_size = encode_cat(info.get("size"), self.plant_vocabs["size"])
        p_climate = encode_cat(care.get("climate"), self.plant_vocabs["climate"])
        p_temp_mid, p_temp_width = temp_features(
            temp_req.get("min_temp"),
            temp_req.get("max_temp"),
        )
        desc_emb = plant.get("desc_embeddings") or [0.0] * DESC_EMBED_DIM

        return {
            "u_cat": torch.tensor([u_light, u_hum, u_care, u_size, u_climate, u_water, u_care_freq], dtype=torch.long),
            "u_num": torch.tensor([u_temp_mid, u_temp_width], dtype=torch.float32),
            "p_cat": torch.tensor([p_light, p_tolerated, p_hum, p_water, p_care, p_size, p_climate], dtype=torch.long),
            "p_num": torch.tensor([p_temp_mid, p_temp_width], dtype=torch.float32),
            "p_desc": torch.tensor(desc_emb[:DESC_EMBED_DIM], dtype=torch.float32),
            "label": torch.tensor(float(label), dtype=torch.float32),
        }


class UserTower(nn.Module):
    def __init__(self, vocabs: dict, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM, out_dim: int = OUTPUT_DIM):
        super().__init__()
        self.embed_light = nn.Embedding(len(LIGHT_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_hum = nn.Embedding(len(HUMIDITY_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_care = nn.Embedding(len(CARE_LEVEL_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_size = nn.Embedding(len(SIZE_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_climate = nn.Embedding(len(CLIMATE_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_water = nn.Embedding(len(WATER_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_care_freq = nn.Embedding(len(WATER_VOCAB) + 1, embed_dim, padding_idx=0)

        cat_dim = 7 * embed_dim
        num_dim = 2
        in_dim = cat_dim + num_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, u_cat: torch.Tensor, u_num: torch.Tensor) -> torch.Tensor:
        # u_cat: (B, 7), u_num: (B, 2)
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


class PlantTower(nn.Module):
    def __init__(self, vocabs: dict, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM, out_dim: int = OUTPUT_DIM):
        super().__init__()
        self.embed_light = nn.Embedding(len(LIGHT_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_tolerated = nn.Embedding(len(LIGHT_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_hum = nn.Embedding(len(HUMIDITY_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_water = nn.Embedding(len(WATER_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_care = nn.Embedding(len(CARE_LEVEL_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_size = nn.Embedding(len(SIZE_VOCAB) + 1, embed_dim, padding_idx=0)
        self.embed_climate = nn.Embedding(len(CLIMATE_VOCAB) + 1, embed_dim, padding_idx=0)

        self.desc_proj = nn.Linear(DESC_EMBED_DIM, 64)  # project Voyage 1024 -> 64
        self.desc_dropout = nn.Dropout(DROPOUT)

        cat_dim = 7 * embed_dim
        num_dim = 2
        desc_dim = 64
        in_dim = cat_dim + num_dim + desc_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, p_cat: torch.Tensor, p_num: torch.Tensor, p_desc: torch.Tensor) -> torch.Tensor:
        e_light = self.embed_light(p_cat[:, 0])
        e_tolerated = self.embed_tolerated(p_cat[:, 1])
        e_hum = self.embed_hum(p_cat[:, 2])
        e_water = self.embed_water(p_cat[:, 3])
        e_care = self.embed_care(p_cat[:, 4])
        e_size = self.embed_size(p_cat[:, 5])
        e_climate = self.embed_climate(p_cat[:, 6])
        cat = torch.cat([e_light, e_tolerated, e_hum, e_water, e_care, e_size, e_climate], dim=1)
        # L2-normalize Voyage vectors (safe for zero vectors)
        p_desc_norm = p_desc / (p_desc.norm(dim=1, keepdim=True) + 1e-8)
        desc = self.desc_dropout(self.desc_proj(p_desc_norm))
        x = torch.cat([cat, p_num, desc], dim=1)
        return self.mlp(x)


class TwoTowerModel(nn.Module):
    def __init__(self, user_vocabs: dict, plant_vocabs: dict):
        super().__init__()
        self.user_tower = UserTower(user_vocabs)
        self.plant_tower = PlantTower(plant_vocabs)

    def forward(self, u_cat, u_num, p_cat, p_num, p_desc):
        u_emb = self.user_tower(u_cat, u_num)
        p_emb = self.plant_tower(p_cat, p_num, p_desc)
        u_emb = F.normalize(u_emb, dim=1)
        p_emb = F.normalize(p_emb, dim=1)
        return (u_emb * p_emb).sum(dim=1) / TAU


def _build_user_features(user: dict, user_vocabs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (u_cat, u_num) for a single user."""
    env = user.get("environment", {}) or {}
    pref = user.get("preferences", {}) or {}
    care_pref = pref.get("care_preferences", {}) or {}
    constraints = user.get("constraints", {}) or {}
    temp = env.get("temperature_pref", {}) or {}
    u_cat = torch.tensor([[
        encode_cat(env.get("light_level"), user_vocabs["light"]),
        encode_cat(env.get("humidity_level"), user_vocabs["humidity"]),
        encode_cat(pref.get("care_level"), user_vocabs["care_level"]),
        encode_cat(constraints.get("preferred_size"), user_vocabs["size"]),
        encode_cat(user.get("climate"), user_vocabs["climate"]),
        encode_cat(care_pref.get("watering_freq"), user_vocabs["watering"]),
        encode_cat(care_pref.get("care_freq"), user_vocabs["care_freq"]),
    ]], dtype=torch.long)
    u_temp_mid, u_temp_width = temp_features(temp.get("min_f"), temp.get("max_f"))
    u_num = torch.tensor([[u_temp_mid, u_temp_width]], dtype=torch.float32)
    return u_cat, u_num


def _build_plant_features(plant: dict, plant_vocabs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (p_cat, p_num, p_desc) for a single plant."""
    care = plant.get("Care", {}) or {}
    info = plant.get("Info", {}) or {}
    light_req = care.get("light_req", {}) or {}
    ideal = light_req.get("ideal_light", {}) or {}
    tolerated = light_req.get("tolerated_light", {}) or {}
    temp_req = care.get("temp_req", {}) or {}
    p_cat = torch.tensor([[
        encode_cat(ideal.get("sunlight_bucket"), plant_vocabs["light"]),
        encode_cat(tolerated.get("sunlight_bucket"), plant_vocabs["light"]),
        encode_cat(care.get("humidity_req_bucket"), plant_vocabs["humidity"]),
        encode_cat(care.get("water_req_bucket"), plant_vocabs["water"]),
        encode_cat(care.get("care_level"), plant_vocabs["care_level"]),
        encode_cat(info.get("size"), plant_vocabs["size"]),
        encode_cat(care.get("climate"), plant_vocabs["climate"]),
    ]], dtype=torch.long)
    p_temp_mid, p_temp_width = temp_features(temp_req.get("min_temp"), temp_req.get("max_temp"))
    p_num = torch.tensor([[p_temp_mid, p_temp_width]], dtype=torch.float32)
    desc = plant.get("desc_embeddings") or [0.0] * DESC_EMBED_DIM
    p_desc = torch.tensor([desc[:DESC_EMBED_DIM]], dtype=torch.float32)
    return p_cat, p_num, p_desc


def _get_val_positives(val_interactions, users, plants, use_oracle_gt: bool):
    """Build val_positives per user.
    If use_oracle_gt: positives = all plants with oracle score >= threshold (or top 50).
    Else: positives = label==1 from val_interactions (~20 sampled).
    """
    val_user_ids = sorted({r["user_id"] for r in val_interactions})
    val_positives = {uid: set() for uid in val_user_ids}

    if use_oracle_gt:
        plant_ids = list(plants.keys())
        for uid in val_user_ids:
            user = users[uid]
            scored = [(pid, compute_oracle_score(user, plants[pid])) for pid in plant_ids]
            scored.sort(key=lambda x: x[1], reverse=True)
            # Positives: score >= threshold, or top N if fewer
            pos_pids = [pid for pid, s in scored if s >= ORACLE_POS_THRESHOLD]
            if len(pos_pids) < 5:  # fallback: top N by score
                pos_pids = [pid for pid, _ in scored[:ORACLE_TOP_N_POSITIVES]]
            val_positives[uid] = set(pos_pids)
    else:
        for r in val_interactions:
            if r["label"] == 1:
                val_positives[r["user_id"]].add(r["plant_id"])

    return val_user_ids, val_positives


def _compute_retrieval_metrics(
    val_user_ids, val_positives, plant_ids, rank_fn,
) -> dict[str, float]:
    """rank_fn(uid) -> list of plant_ids in ranked order (top first)."""
    ks = EVAL_KS
    recall = {k: 0.0 for k in ks}
    precision = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    hit_rate_10 = 0.0
    n_users = 0

    for uid in val_user_ids:
        pos = val_positives[uid]
        if not pos:
            continue
        n_users += 1
        top_pids = rank_fn(uid)

        for k in ks:
            top_k = top_pids[:k]
            hits = sum(1 for pid in top_k if pid in pos)
            recall[k] += hits / len(pos)
            precision[k] += hits / k  # P@K = hits / K
            dcg = sum(1.0 / math.log2(i + 2) for i, pid in enumerate(top_k) if pid in pos)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(pos))))
            ndcg[k] += dcg / idcg if idcg > 0 else 0.0

        hit_rate_10 += 1.0 if any(pid in pos for pid in top_pids[:10]) else 0.0

    if n_users == 0:
        return (
            {f"R@{k}": 0.0 for k in ks}
            | {f"P@{k}": 0.0 for k in ks}
            | {f"NDCG@{k}": 0.0 for k in ks}
            | {"HR@10": 0.0}
        )

    return {
        **{f"R@{k}": recall[k] / n_users for k in ks},
        **{f"P@{k}": precision[k] / n_users for k in ks},
        **{f"NDCG@{k}": ndcg[k] / n_users for k in ks},
        "HR@10": hit_rate_10 / n_users,
    }


def eval_retrieval(
    model, users, plants, plant_vocabs, user_vocabs, val_interactions, device,
    use_oracle_gt: bool = False,
) -> dict[str, float]:
    """Compute Recall@K, NDCG@K, HitRate@10 on held-out users.
    use_oracle_gt: if True, positives = plants with oracle score >= threshold (less sampling-dependent).
    """
    val_user_ids, val_positives = _get_val_positives(val_interactions, users, plants, use_oracle_gt)
    plant_ids = list(plants.keys())
    features = [_build_plant_features(plants[pid], plant_vocabs) for pid in plant_ids]
    p_cats = torch.cat([f[0] for f in features], dim=0)
    p_nums = torch.cat([f[1] for f in features], dim=0)
    p_descs = torch.cat([f[2] for f in features], dim=0)
    p_cat_all = p_cats.to(device)
    p_num_all = p_nums.to(device)
    p_desc_all = p_descs.to(device)

    with torch.no_grad():
        p_emb_all = model.plant_tower(p_cat_all, p_num_all, p_desc_all)
        p_emb_all = F.normalize(p_emb_all, dim=1)

    def rank_fn(uid):
        u_cat, u_num = _build_user_features(users[uid], user_vocabs)
        with torch.no_grad():
            u_emb = model.user_tower(u_cat.to(device), u_num.to(device))
            u_emb = F.normalize(u_emb, dim=1)
            scores = (u_emb @ p_emb_all.T).squeeze(0) / TAU
        _, top_idx = torch.topk(scores, min(max(EVAL_KS), len(plant_ids)))
        return [plant_ids[i] for i in top_idx.cpu().tolist()]

    return _compute_retrieval_metrics(val_user_ids, val_positives, plant_ids, rank_fn)


def eval_oracle_retrieval(users, plants, val_interactions, use_oracle_gt: bool = False) -> dict[str, float]:
    """Oracle upper bound: rank plants by compute_oracle_score. Compare to model metrics."""
    val_user_ids, val_positives = _get_val_positives(val_interactions, users, plants, use_oracle_gt)
    plant_ids = list(plants.keys())

    def rank_fn(uid):
        scored = [(pid, compute_oracle_score(users[uid], plants[pid])) for pid in plant_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in scored]

    return _compute_retrieval_metrics(val_user_ids, val_positives, plant_ids, rank_fn)


def collate_fn(batch):
    return {
        "u_cat": torch.stack([b["u_cat"] for b in batch]),
        "u_num": torch.stack([b["u_num"] for b in batch]),
        "p_cat": torch.stack([b["p_cat"] for b in batch]),
        "p_num": torch.stack([b["p_num"] for b in batch]),
        "p_desc": torch.stack([b["p_desc"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def main():
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(USERS_PATH) as f:
        users_raw = json.load(f)
    with open(PLANTS_PATH) as f:
        plants_raw = json.load(f)
    with open(INTERACTIONS_PATH) as f:
        interactions = json.load(f)

    users = {u["user_id"]: u for u in users_raw}
    plants = {p["plant_id"]: p for p in plants_raw}

    user_vocabs = {
        "light": vocab_to_idx(LIGHT_VOCAB),
        "humidity": vocab_to_idx(HUMIDITY_VOCAB),
        "care_level": vocab_to_idx(CARE_LEVEL_VOCAB),
        "size": vocab_to_idx(SIZE_VOCAB),
        "climate": vocab_to_idx(CLIMATE_VOCAB),
        "watering": vocab_to_idx(WATER_VOCAB),
        "care_freq": vocab_to_idx(WATER_VOCAB),
    }
    plant_vocabs = {
        "light": vocab_to_idx(LIGHT_VOCAB),
        "humidity": vocab_to_idx(HUMIDITY_VOCAB),
        "water": vocab_to_idx(WATER_VOCAB),
        "care_level": vocab_to_idx(CARE_LEVEL_VOCAB),
        "size": vocab_to_idx(SIZE_VOCAB),
        "climate": vocab_to_idx(CLIMATE_VOCAB),
    }

    # Train/val split by user (no leakage)
    user_ids = sorted({r["user_id"] for r in interactions})
    random.shuffle(user_ids)
    n_val_users = int(len(user_ids) * VAL_RATIO)
    val_users = set(user_ids[:n_val_users])

    train_interactions = [r for r in interactions if r["user_id"] not in val_users]
    val_interactions = [r for r in interactions if r["user_id"] in val_users]

    train_pos = sum(1 for r in train_interactions if r["label"] == 1)
    train_neg = sum(1 for r in train_interactions if r["label"] == 0)
    pos_weight = torch.tensor([train_neg / train_pos] if train_pos > 0 else [1.0])

    # Oracle upper bound (sampled positives)
    oracle_metrics = eval_oracle_retrieval(users, plants, val_interactions, use_oracle_gt=False)
    print("Oracle upper bound (sampled positives):", end=" ")
    print(" ".join(f"{k}={v:.3f}" for k, v in oracle_metrics.items()))

    train_ds = UserPlantDataset(train_interactions, users, plants, user_vocabs, plant_vocabs)
    val_ds = UserPlantDataset(val_interactions, users, plants, user_vocabs, plant_vocabs)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = TwoTowerModel(user_vocabs, plant_vocabs)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    model = model.to(device)

    best_r10 = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            u_cat = batch["u_cat"].to(device)
            u_num = batch["u_num"].to(device)
            p_cat = batch["p_cat"].to(device)
            p_num = batch["p_num"].to(device)
            p_desc = batch["p_desc"].to(device)
            label = batch["label"].to(device)

            opt.zero_grad()
            logits = model(u_cat, u_num, p_cat, p_num, p_desc)
            loss = bce(logits, label)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                u_cat = batch["u_cat"].to(device)
                u_num = batch["u_num"].to(device)
                p_cat = batch["p_cat"].to(device)
                p_num = batch["p_num"].to(device)
                p_desc = batch["p_desc"].to(device)
                label = batch["label"].to(device)

                logits = model(u_cat, u_num, p_cat, p_num, p_desc)
                val_loss += bce(logits, label).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        metrics = eval_retrieval(
            model, users, plants, plant_vocabs, user_vocabs, val_interactions, device,
            use_oracle_gt=False,
        )
        metric_str = " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  {metric_str}")

        # Early stopping: save best by R@10
        r10 = metrics.get("R@10", 0.0)
        if r10 > best_r10:
            best_r10 = r10
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "two_tower_best.pt")
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (no R@10 improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # Restore best checkpoint
    if (OUTPUT_DIR / "two_tower_best.pt").exists():
        model.load_state_dict(torch.load(OUTPUT_DIR / "two_tower_best.pt", map_location=device))
        print(f"Restored best checkpoint from epoch {best_epoch} (R@10={best_r10:.4f})")

    # Eval with oracle-defined ground truth (less sampling-dependent)
    oracle_gt_metrics = eval_retrieval(
        model, users, plants, plant_vocabs, user_vocabs, val_interactions, device,
        use_oracle_gt=True,
    )
    oracle_gt_oracle = eval_oracle_retrieval(users, plants, val_interactions, use_oracle_gt=True)
    print("Eval with oracle ground truth (score>=threshold):")
    print("  Model:  ", " ".join(f"{k}={v:.3f}" for k, v in oracle_gt_metrics.items()))
    print("  Oracle:", " ".join(f"{k}={v:.3f}" for k, v in oracle_gt_oracle.items()))

    # Save model and plant embeddings for inference
    torch.save(model.state_dict(), OUTPUT_DIR / "two_tower.pt")
    if (OUTPUT_DIR / "two_tower_best.pt").exists():
        (OUTPUT_DIR / "two_tower_best.pt").unlink()  # remove redundant checkpoint
    model.eval()
    plant_embeds = []
    for plant in plants_raw:
        pid = plant["plant_id"]
        p_cat, p_num, p_desc = _build_plant_features(plant, plant_vocabs)
        with torch.no_grad():
            emb = model.plant_tower(p_cat.to(device), p_num.to(device), p_desc.to(device))
            emb = F.normalize(emb, dim=1)
        plant_embeds.append({"plant_id": pid, "embedding": emb.cpu().numpy()[0].tolist()})

    with open(OUTPUT_DIR / "plant_embeddings.json", "w") as f:
        json.dump(plant_embeds, f, indent=2)

    print(f"Saved model -> {OUTPUT_DIR / 'two_tower.pt'}")
    print(f"Saved plant embeddings (L2-normalized) -> {OUTPUT_DIR / 'plant_embeddings.json'}")
    print("At inference: normalize user embeddings with F.normalize(u_emb, dim=1) before dot product.")


if __name__ == "__main__":
    main()
