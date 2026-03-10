"""
Retrain two-tower model from scratch on synthetic interactions (and optionally real from MongoDB).
Saves model, retrain_metrics.txt, and plant_embeddings.json. Optionally updates MongoDB PlantCollection with new plant_tower_embedding.

Weights (real data prioritized):
  synthetic: 0.3
  real garden >30d: 1.0, recent: 2.0
  deaths: 4.0

When real data exists: synthetic is subsampled to max(real × 3, 5000) to reduce dominance.
Death feedback eval: measures if killed plants drop in rank (lower = better).
Retrain eval uses model scoring only (no death penalty). Death penalty is in recommend pipeline only.

Run from project root:
  python -m backend.recommend.retrain.retrain_two_tower                    # synthetic + real, update MongoDB (default)
  python -m backend.recommend.retrain.retrain_two_tower --no-update-mongo # skip MongoDB update
  python -m backend.recommend.retrain.retrain_two_tower --no-include-real  # synthetic only
  python -m backend.recommend.retrain.retrain_two_tower --dvc-add          # track outputs with DVC
"""
from __future__ import annotations

import cProfile  # noqa: F401 - load stdlib before backend.profile shadows it
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Project root and backend on path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Import training components from resources/two_tower_training (not on default path)
import importlib.util
_ttt_path = ROOT / "resources" / "two_tower_training" / "two_tower_training.py"
_spec = importlib.util.spec_from_file_location("two_tower_training", _ttt_path)
ttt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ttt)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Paths
USERS_PATH = ROOT / "resources" / "synthetic_users.json"
PLANTS_PATH = ROOT / "resources" / "plant_profiles.json"
INTERACTIONS_PATH = ROOT / "resources" / "interactions.json"
# Fallback: data_creating if interactions generated there
INTERACTIONS_PATH_ALT = ROOT / "resources" / "data_creating" / "interactions.json"
OUTPUT_DIR = ROOT / "resources" / "two_tower_training" / "output"

# Retrain config (fewer epochs for fine-tune, lower LR)
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "5"))
RETRAIN_LR = float(os.getenv("RETRAIN_LR", "5e-4"))

# Interaction weights: real data weighted higher to reduce synthetic dominance
WEIGHT_OLDER = float(os.getenv("RETRAIN_WEIGHT_OLDER", "0.3"))       # synthetic only
WEIGHT_REAL_OLDER = float(os.getenv("RETRAIN_WEIGHT_REAL_OLDER", "1.0"))  # real garden >30d
WEIGHT_RECENT = float(os.getenv("RETRAIN_WEIGHT_RECENT", "2.0"))     # real garden recent
WEIGHT_DEATH = float(os.getenv("RETRAIN_WEIGHT_DEATH", "4.0"))       # deaths
RECENT_DAYS = int(os.getenv("RETRAIN_RECENT_DAYS", "30"))
N_NEGATIVES_PER_POSITIVE = int(os.getenv("RETRAIN_N_NEGATIVES_PER_POSITIVE", "4"))

# Synthetic subsampling: when real data exists, cap synthetic to avoid dominance
# max_synthetic = real_count * SYNT_MAX_RATIO, at least SYNT_MIN_CAP
SYNT_MAX_RATIO = float(os.getenv("RETRAIN_SYNT_MAX_RATIO", "3.0"))  # synthetic <= 3x real
SYNT_MIN_CAP = int(os.getenv("RETRAIN_SYNT_MIN_CAP", "5000"))       # keep at least 5k synthetic


def _mongo_user_to_profile(doc: dict) -> dict:
    """Convert MongoDB user doc to synthetic-user-style profile for training."""
    env = doc.get("environment") or {}
    pref = doc.get("preferences") or {}
    constraints = doc.get("constraints") or {}
    return {
        "environment": env,
        "preferences": pref,
        "constraints": constraints,
        "climate": doc.get("climate"),
    }


def _load_real_interactions() -> tuple[list[dict], dict[int, dict]]:
    """
    Load real interactions from MongoDB: garden adds (positive), deaths (negative),
    and sampled negatives for users with positives.
    Returns (interactions, users_by_id). users_by_id maps synthetic user_id -> profile.
    """
    try:
        from database import get_death_collection, get_garden_collection, get_user_collection
    except ImportError:
        print("  MongoDB not available. Skipping real interactions.")
        return [], {}, []

    user_coll = get_user_collection()
    garden_coll = get_garden_collection()
    death_coll = get_death_collection()

    # Load plant catalog for negative sampling
    with open(PLANTS_PATH) as f:
        plants_raw = json.load(f)
    all_plant_ids = [p["plant_id"] for p in plants_raw]

    # username -> user_id (assign IDs above synthetic max)
    with open(USERS_PATH) as f:
        syn_users = json.load(f)
    max_syn_id = max(u["user_id"] for u in syn_users) if syn_users else -1
    username_to_id: dict[str, int] = {}
    next_id = max_syn_id + 1

    def get_user_id(username: str) -> int:
        nonlocal next_id
        if username not in username_to_id:
            username_to_id[username] = next_id
            next_id += 1
        return username_to_id[username]

    # Per-user: positive pids (garden), death pids
    user_positive_pids: dict[str, set[int]] = {}
    user_positive_weights: dict[str, dict[int, float]] = {}  # username -> {plant_id: weight}
    user_death_pids: dict[str, set[int]] = {}

    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=RECENT_DAYS)

    # Positives: garden adds (weight 1.0 if recent, 0.5 if older)
    n_garden = 0
    for doc in garden_coll.find({}, {"username": 1, "plant_id": 1, "added_at": 1}):
        username = doc.get("username")
        plant_id = doc.get("plant_id")
        if username is None or plant_id is None:
            continue
        get_user_id(username)
        added_at = doc.get("added_at")
        if added_at:
            added_dt = added_at if added_at.tzinfo else added_at.replace(tzinfo=timezone.utc)
            weight = WEIGHT_RECENT if added_dt >= recent_cutoff else WEIGHT_REAL_OLDER
        else:
            weight = WEIGHT_REAL_OLDER
        user_positive_pids.setdefault(username, set()).add(plant_id)
        user_positive_weights.setdefault(username, {})[plant_id] = weight
        n_garden += 1
    print(f"  Retrieved {n_garden} plants from garden")

    # Deaths: explicit negatives (load all; mark used_in_retraining=true after retrain to stop death penalty)
    death_records_used: list[tuple[str, int]] = []
    n_deaths = 0
    for doc in death_coll.find({}, {"username": 1, "plant_id": 1}):
        username = doc.get("username")
        plant_id = doc.get("plant_id")
        if username is None or plant_id is None:
            continue
        get_user_id(username)
        user_death_pids.setdefault(username, set()).add(plant_id)
        death_records_used.append((username, plant_id))
        n_deaths += 1
    print(f"  Retrieved {n_deaths} death records")

    # Build interactions: positives, deaths, and sampled negatives
    interactions: list[dict] = []

    for username in set(user_positive_pids.keys()) | set(user_death_pids.keys()):
        uid = username_to_id.get(username)
        if uid is None:
            continue
        pos_pids = user_positive_pids.get(username, set())
        death_pids = user_death_pids.get(username, set())
        weights = user_positive_weights.get(username, {})

        # Add positives
        for pid in pos_pids:
            interactions.append({
                "user_id": uid, "plant_id": pid, "label": 1, "score": 1.0,
                "weight": weights.get(pid, WEIGHT_OLDER),
            })

        # Add deaths (explicit negatives)
        for pid in death_pids:
            interactions.append({
                "user_id": uid, "plant_id": pid, "label": 0, "score": 0.0,
                "weight": WEIGHT_DEATH,
            })

        # Sampled negatives: for users with positives, sample plants not in pos or death
        if pos_pids:
            excluded = pos_pids | death_pids
            sample_from = [p for p in all_plant_ids if p not in excluded]
            n_to_sample = min(len(pos_pids) * N_NEGATIVES_PER_POSITIVE, len(sample_from))
            if n_to_sample > 0 and sample_from:
                sampled = random.sample(sample_from, n_to_sample)
                for pid in sampled:
                    interactions.append({
                        "user_id": uid, "plant_id": pid, "label": 0, "score": 0.0,
                        "weight": WEIGHT_OLDER,
                    })

    if not interactions:
        return [], {}, []

    n_pos = sum(1 for r in interactions if r["label"] == 1)
    n_neg = sum(1 for r in interactions if r["label"] == 0)
    print(f"  Built {len(interactions)} real interactions ({n_pos} positives, {n_neg} negatives) from {len(username_to_id)} users")

    # Build users dict: fetch profiles for each username
    users_by_id: dict[int, dict] = {}
    for username, uid in username_to_id.items():
        user_doc = user_coll.find_one({"auth.username": username}) or user_coll.find_one(
            {"auth.email": username.lower()}
        )
        if not user_doc:
            continue
        profile = _mongo_user_to_profile(user_doc)
        profile["user_id"] = uid
        profile["climate"] = user_doc.get("climate")
        users_by_id[uid] = profile

    # Filter interactions to users we have profiles for
    valid_uids = set(users_by_id.keys())
    interactions = [r for r in interactions if r["user_id"] in valid_uids]
    # Keep only death_records_used for users we have profiles for
    valid_usernames = {u for u, uid in username_to_id.items() if uid in valid_uids}
    death_records_used = [(u, p) for u, p in death_records_used if u in valid_usernames]
    dropped = len(username_to_id) - len(valid_uids)
    if dropped > 0:
        print(f"  Dropped interactions for {dropped} users without profiles")
    return interactions, users_by_id, death_records_used


def _load_merged_interactions(include_real: bool) -> tuple[list[dict], dict[int, dict], dict[int, dict], list[tuple[int, int]], list[tuple[str, int]]]:
    """Load synthetic + optional real interactions. Returns (interactions, users, plants, death_pairs, death_records_used)."""
    interactions_path = INTERACTIONS_PATH if INTERACTIONS_PATH.exists() else INTERACTIONS_PATH_ALT
    if not interactions_path.exists():
        raise FileNotFoundError(
            f"Interactions not found. Run: python -m resources.data_creating.interactions "
            f"(or ensure {INTERACTIONS_PATH} or {INTERACTIONS_PATH_ALT} exists)"
        )

    print("  Loading synthetic interactions...")
    with open(interactions_path) as f:
        interactions = json.load(f)
    with open(USERS_PATH) as f:
        users_raw = json.load(f)
    with open(PLANTS_PATH) as f:
        plants_raw = json.load(f)

    users = {u["user_id"]: u for u in users_raw}
    plants = {p["plant_id"]: p for p in plants_raw}
    print(f"  Loaded {len(interactions)} synthetic interactions, {len(users)} users, {len(plants)} plants")

    # Synthetic = lower weight to reduce dominance
    for r in interactions:
        r.setdefault("weight", WEIGHT_OLDER)

    death_records_used: list[tuple[str, int]] = []
    real_interactions: list[dict] = []
    if include_real:
        print("  Loading real interactions from MongoDB...")
        real_interactions, real_users, death_records_used = _load_real_interactions()
        if not real_interactions:
            print("  No real interactions found in MongoDB (empty garden/death collections)")
            return interactions, users, plants, [], []
        elif real_interactions:
            # Filter to plant_ids that exist in catalog
            valid_pids = set(plants.keys())
            real_interactions = [r for r in real_interactions if r["plant_id"] in valid_pids]
            n_real = len(real_interactions)

            # Subsample synthetic to reduce dominance (keep real data influential)
            max_syn = max(int(n_real * SYNT_MAX_RATIO), SYNT_MIN_CAP)
            if len(interactions) > max_syn:
                interactions = random.sample(interactions, max_syn)
                print(f"  Subsampled synthetic to {len(interactions)} (max {max_syn} = {n_real} real × {SYNT_MAX_RATIO})")

            # Real interactions first (priority), then synthetic
            interactions = real_interactions + interactions
            users.update(real_users)
            n_syn = len(interactions) - n_real
            w_death = sum(1 for r in real_interactions if r.get("weight") == WEIGHT_DEATH)
            w_recent = sum(1 for r in real_interactions if r.get("weight") == WEIGHT_RECENT)
            w_real_older = sum(1 for r in real_interactions if r.get("weight") == WEIGHT_REAL_OLDER)
            n_sampled_neg = sum(1 for r in real_interactions if r["label"] == 0 and r.get("weight") == WEIGHT_OLDER)
            print(f"  Merged: {n_real} real (deaths={w_death}, sampled_neg={n_sampled_neg}, recent_pos={w_recent}, older_pos={w_real_older}) + {n_syn} synthetic")

    # Death pairs for eval: (user_id, plant_id) - only from real deaths
    death_pairs: list[tuple[int, int]] = []
    if include_real and real_interactions:
        death_pairs = [(r["user_id"], r["plant_id"]) for r in real_interactions if r["label"] == 0 and r.get("weight") == WEIGHT_DEATH]

    return interactions, users, plants, death_pairs, death_records_used


class WeightedUserPlantDataset(ttt.UserPlantDataset):
    """UserPlantDataset that adds sample weight from interaction['weight']."""

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        row = self.interactions[idx]
        out["weight"] = torch.tensor(float(row.get("weight", 1.0)), dtype=torch.float32)
        return out


def _collate_fn_weighted(batch):
    """Collate that includes weight tensor."""
    base = ttt.collate_fn(batch)
    base["weight"] = torch.stack([b["weight"] for b in batch])
    return base


def _eval_death_feedback(
    model: torch.nn.Module,
    users: dict[int, dict],
    plants: dict[int, dict],
    plant_vocabs: dict,
    user_vocabs: dict,
    death_pairs: list[tuple[int, int]],
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate death feedback: for users who reported a plant death, measure if that plant
    drops in rank. Lower rank = better (model learned to deprioritize killed plants).
    """
    if not death_pairs:
        return {}

    TAU = getattr(ttt, "TAU", 0.1)
    plant_ids = list(plants.keys())
    valid_pairs = [(uid, pid) for uid, pid in death_pairs if uid in users and pid in plants and pid in plant_ids]
    if not valid_pairs:
        return {}

    # Precompute plant embeddings
    features = [ttt._build_plant_features(plants[pid], plant_vocabs) for pid in plant_ids]
    p_cats = torch.cat([f[0] for f in features], dim=0)
    p_nums = torch.cat([f[1] for f in features], dim=0)
    p_descs = torch.cat([f[2] for f in features], dim=0)
    with torch.no_grad():
        p_emb_all = model.plant_tower(p_cats.to(device), p_nums.to(device), p_descs.to(device))
        p_emb_all = F.normalize(p_emb_all, dim=1)

    ranks = []
    for uid, killed_pid in valid_pairs:
        u_cat, u_num = ttt._build_user_features(users[uid], user_vocabs)
        with torch.no_grad():
            u_emb = model.user_tower(u_cat.to(device), u_num.to(device))
            u_emb = F.normalize(u_emb, dim=1)
            scores = (u_emb @ p_emb_all.T).squeeze(0) / TAU
        _, top_idx = torch.topk(scores, len(plant_ids))
        ranked_pids = [plant_ids[i] for i in top_idx.cpu().tolist()]
        try:
            rank = ranked_pids.index(killed_pid) + 1  # 1-indexed
        except ValueError:
            rank = len(plant_ids) + 1
        ranks.append(rank)

    n = len(ranks)
    avg_rank = sum(ranks) / n
    below_20 = sum(1 for r in ranks if r > 20) / n * 100  # % pushed below top 20
    below_50 = sum(1 for r in ranks if r > 50) / n * 100
    median_rank = sorted(ranks)[n // 2] if n else 0
    return {
        "death_avg_rank": round(avg_rank, 1),
        "death_median_rank": median_rank,
        "death_pct_below_rank20": round(below_20, 1),
        "death_pct_below_rank50": round(below_50, 1),
        "death_n_pairs": n,
    }


def _update_mongo_embeddings(plant_embeddings: list[dict]) -> int:
    """Update PlantCollection with new plant_tower_embedding. Returns count updated."""
    try:
        from database import get_plant_collection
    except ImportError:
        print("MongoDB not available. Skip update.")
        return 0

    coll = get_plant_collection()
    emb_by_id = {p["plant_id"]: p["embedding"] for p in plant_embeddings}
    updated = 0
    for pid, emb in emb_by_id.items():
        result = coll.update_one(
            {"plant_id": pid},
            {"$set": {"plant_tower_embedding": emb}},
        )
        if result.modified_count:
            updated += 1
    return updated


def _mark_deaths_used_in_retraining(death_records: list[tuple[str, int]]) -> int:
    """Mark death records as used_in_retraining=True after successful retrain. Returns count updated."""
    if not death_records:
        return 0
    try:
        from database import get_death_collection
    except ImportError:
        print("MongoDB not available. Skip marking deaths used.")
        return 0

    death_coll = get_death_collection()
    updated = 0
    for username, plant_id in death_records:
        result = death_coll.update_many(
            {"username": username, "plant_id": plant_id},
            {"$set": {"used_in_retraining": True}},
        )
        updated += result.modified_count
    return updated


def main():
    include_real = "--no-include-real" not in sys.argv  # default True
    update_mongo = "--no-update-mongo" not in sys.argv  # default True
    dvc_add = "--dvc-add" in sys.argv

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if include_real:
        print("Mode: synthetic + real (MongoDB garden/death)")
    else:
        print("Mode: synthetic only (pass --no-include-real to skip real garden/death data)")

    print("Loading data...")
    interactions, users, plants, death_pairs, death_records_used = _load_merged_interactions(include_real)
    # Count real vs synthetic users (real get user_id > max synthetic)
    with open(USERS_PATH) as f:
        syn_user_ids = {u["user_id"] for u in json.load(f) if "user_id" in u}
    max_syn_uid = max(syn_user_ids) if syn_user_ids else -1
    n_real_users = sum(1 for uid in users if uid > max_syn_uid)
    n_syn_users = len(users) - n_real_users
    if include_real and n_real_users > 0:
        print(f"  {len(interactions)} interactions, {len(users)} users ({n_real_users} real, {n_syn_users} synthetic), {len(plants)} plants")
    else:
        print(f"  {len(interactions)} interactions, {len(users)} users, {len(plants)} plants")

    if not interactions:
        print("No interactions. Run resources/data_creating/interactions.py first.")
        sys.exit(1)

    user_vocabs = {
        "light": ttt.vocab_to_idx(ttt.LIGHT_VOCAB),
        "humidity": ttt.vocab_to_idx(ttt.HUMIDITY_VOCAB),
        "care_level": ttt.vocab_to_idx(ttt.CARE_LEVEL_VOCAB),
        "size": ttt.vocab_to_idx(ttt.SIZE_VOCAB),
        "climate": ttt.vocab_to_idx(ttt.CLIMATE_VOCAB),
        "watering": ttt.vocab_to_idx(ttt.WATER_VOCAB),
        "care_freq": ttt.vocab_to_idx(ttt.WATER_VOCAB),
    }
    plant_vocabs = {
        "light": ttt.vocab_to_idx(ttt.LIGHT_VOCAB),
        "humidity": ttt.vocab_to_idx(ttt.HUMIDITY_VOCAB),
        "water": ttt.vocab_to_idx(ttt.WATER_VOCAB),
        "care_level": ttt.vocab_to_idx(ttt.CARE_LEVEL_VOCAB),
        "size": ttt.vocab_to_idx(ttt.SIZE_VOCAB),
        "climate": ttt.vocab_to_idx(ttt.CLIMATE_VOCAB),
    }

    # Train/val split
    user_ids = sorted({r["user_id"] for r in interactions})
    random.shuffle(user_ids)
    n_val = int(len(user_ids) * ttt.VAL_RATIO)
    val_users = set(user_ids[:n_val])
    train_interactions = [r for r in interactions if r["user_id"] not in val_users]
    val_interactions = [r for r in interactions if r["user_id"] in val_users]
    print(f"  Train/val split: {len(train_interactions)} train, {len(val_interactions)} val interactions")

    train_pos = sum(1 for r in train_interactions if r["label"] == 1)
    train_neg = sum(1 for r in train_interactions if r["label"] == 0)
    pos_weight = torch.tensor([train_neg / train_pos] if train_pos > 0 else [1.0])

    train_ds = WeightedUserPlantDataset(train_interactions, users, plants, user_vocabs, plant_vocabs)
    val_ds = WeightedUserPlantDataset(val_interactions, users, plants, user_vocabs, plant_vocabs)
    train_loader = DataLoader(
        train_ds, batch_size=ttt.BATCH_SIZE, shuffle=True, collate_fn=_collate_fn_weighted, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=ttt.BATCH_SIZE, shuffle=False, collate_fn=_collate_fn_weighted, num_workers=0
    )

    model = ttt.TwoTowerModel(user_vocabs, plant_vocabs)
    print("Training from scratch")

    opt = torch.optim.Adam(model.parameters(), lr=RETRAIN_LR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
    model = model.to(device)

    plants_raw = list(plants.values())
    best_r10 = 0.0
    best_epoch = 0

    for epoch in range(RETRAIN_EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            u_cat = batch["u_cat"].to(device)
            u_num = batch["u_num"].to(device)
            p_cat = batch["p_cat"].to(device)
            p_num = batch["p_num"].to(device)
            p_desc = batch["p_desc"].to(device)
            label = batch["label"].to(device)
            weight = batch["weight"].to(device)
            opt.zero_grad()
            logits = model(u_cat, u_num, p_cat, p_num, p_desc)
            loss_per_sample = bce(logits, label)
            loss = (weight * loss_per_sample).mean()
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
                weight = batch["weight"].to(device)
                logits = model(u_cat, u_num, p_cat, p_num, p_desc)
                loss_per_sample = bce(logits, label)
                val_loss += (weight * loss_per_sample).mean().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        metrics = ttt.eval_retrieval(
            model, users, plants, plant_vocabs, user_vocabs, val_interactions, device, use_oracle_gt=False
        )
        r10 = metrics.get("R@10", 0.0)
        print(
            f"Epoch {epoch+1}/{RETRAIN_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"R@10={r10:.3f}  NDCG@10={metrics.get('NDCG@10', 0):.3f}"
        )
        if r10 > best_r10:
            best_r10 = r10
            best_epoch = epoch + 1
            torch.save(model.state_dict(), OUTPUT_DIR / "two_tower.pt")

    print(f"Best R@10={best_r10:.4f} at epoch {best_epoch}")

    # Final metrics at best epoch
    model.load_state_dict(torch.load(OUTPUT_DIR / "two_tower.pt", map_location=device))
    model.eval()
    final_metrics = ttt.eval_retrieval(
        model, users, plants, plant_vocabs, user_vocabs, val_interactions, device, use_oracle_gt=False
    )

    # Death feedback evaluation: do killed plants drop in rank?
    death_metrics = None
    if death_pairs:
        death_metrics = _eval_death_feedback(
            model, users, plants, plant_vocabs, user_vocabs, death_pairs, device
        )
        if death_metrics:
            print(
                f"Death feedback: avg_rank={death_metrics['death_avg_rank']}, median={death_metrics['death_median_rank']}, "
                f"% below rank20={death_metrics['death_pct_below_rank20']}%, below rank50={death_metrics['death_pct_below_rank50']}% "
                f"(n={death_metrics['death_n_pairs']} death pairs)"
            )

    # Write metrics to file
    metrics_lines = [
        f"retrain_metrics {datetime.now(timezone.utc).isoformat()}",
        "",
        "config",
        f"  RETRAIN_EPOCHS={RETRAIN_EPOCHS}",
        f"  RETRAIN_LR={RETRAIN_LR}",
        f"  include_real={include_real}",
        "",
        "data",
        f"  n_train={len(train_interactions)}",
        f"  n_val={len(val_interactions)}",
        f"  n_users={len(users)}",
        f"  n_plants={len(plants)}",
        f"  n_death_pairs={len(death_pairs)}",
        "",
        "retrieval (best epoch)",
        f"  best_epoch={best_epoch}",
    ]
    for k, v in sorted(final_metrics.items()):
        metrics_lines.append(f"  {k}={v:.4f}")
    if death_metrics:
        metrics_lines.extend([
            "",
            "death_feedback",
            f"  death_avg_rank={death_metrics['death_avg_rank']}",
            f"  death_median_rank={death_metrics['death_median_rank']}",
            f"  death_pct_below_rank20={death_metrics['death_pct_below_rank20']}",
            f"  death_pct_below_rank50={death_metrics['death_pct_below_rank50']}",
            f"  death_n_pairs={death_metrics['death_n_pairs']}",
        ])
    metrics_path = OUTPUT_DIR / "retrain_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("\n".join(metrics_lines))
    print(f"Saved metrics to {metrics_path}")

    # Compute plant embeddings
    print("Computing plant embeddings...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "two_tower.pt", map_location=device))
    model.eval()
    plant_embeds = []
    for plant in plants_raw:
        pid = plant["plant_id"]
        p_cat, p_num, p_desc = ttt._build_plant_features(plant, plant_vocabs)
        with torch.no_grad():
            emb = model.plant_tower(p_cat.to(device), p_num.to(device), p_desc.to(device))
            emb = F.normalize(emb, dim=1)
        plant_embeds.append({"plant_id": pid, "embedding": emb.cpu().numpy()[0].tolist()})

    emb_path = OUTPUT_DIR / "plant_embeddings.json"
    with open(emb_path, "w") as f:
        json.dump(plant_embeds, f, indent=2)
    print(f"Saved {len(plant_embeds)} plant embeddings to {emb_path}")

    if update_mongo:
        print("Updating MongoDB PlantCollection...")
        n = _update_mongo_embeddings(plant_embeds)
        print(f"Updated {n} plants in MongoDB PlantCollection")

    if death_records_used:
        n_marked = _mark_deaths_used_in_retraining(death_records_used)
        print(f"Marked {n_marked} death records as used_in_retraining=True")

    if dvc_add:
        print("Adding model and metrics to DVC...")
        for path in [OUTPUT_DIR / "two_tower.pt", OUTPUT_DIR / "retrain_metrics.txt"]:
            subprocess.run(["dvc", "add", str(path)], cwd=str(ROOT), check=True)
        print("DVC add done. Commit .dvc files and run 'dvc push' to upload to remote.")


if __name__ == "__main__":
    main()
