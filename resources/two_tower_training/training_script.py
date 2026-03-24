"""
Train two-tower model on synthetic interactions.
Loads features from Feast (no feature engineering). Trains with BCE loss.
Evaluates with AUC, accuracy, precision, recall, F1, NDCG@k.
After training, pushes plant embeddings to MongoDB.
Run from project root: python -m resources.two_tower_training.training_script
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import torch

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INTERACTIONS_PATH = ROOT / "resources" / "data" / "synthetic_interactions.json"
OUTPUT_DIR = Path(__file__).resolve().parent

BATCH_SIZE = 256
EPOCHS = 15
LR = 1e-3
VAL_RATIO = 0.1
SEED = 42
NDCG_K = [5, 10, 20]


USER_INPUT_DIM = 21 + 64 + 64  # categorical_embedding + plants_grown_agg + plants_killed_agg


class InteractionDataset(Dataset):
    """Dataset of (user_vec, plant_vec, label). user_vec = [categorical_embedding, plants_grown_agg, plants_killed_agg]."""

    def __init__(self, interactions: list[dict], user_by_id: dict, plant_by_id: dict):
        self.samples = []
        for r in interactions:
            uid, pid = r["user_id"], r["plant_id"]
            user = user_by_id.get(uid)
            plant = plant_by_id.get(pid)
            u_emb = user.get("categorical_embedding") if user else None
            grown = user.get("plants_grown_agg", [0.0] * 64) if user else [0.0] * 64
            killed = user.get("plants_killed_agg", [0.0] * 64) if user else [0.0] * 64
            p_emb = plant.get("categorical_embedding") if plant else None
            if u_emb and p_emb and len(u_emb) == 21 and len(p_emb) == 21:
                user_vec = list(u_emb) + list(grown[:64]) + list(killed[:64])
                self.samples.append({
                    "user_vec": user_vec,
                    "plant_vec": p_emb,
                    "label": float(r["label"]),
                    "user_id": uid,
                    "plant_id": pid,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "user_vec": torch.tensor(s["user_vec"], dtype=torch.float32),
            "plant_vec": torch.tensor(s["plant_vec"], dtype=torch.float32),
            "label": torch.tensor(s["label"], dtype=torch.float32),
            "user_id": s["user_id"],
            "plant_id": s["plant_id"],
        }


def _dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted cumulative gain at k. DCG = sum(rel_i / log2(i+2))."""
    relevances = relevances[:k]
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def _ndcg_at_k(relevances: list[float], k: int) -> float:
    """Normalized DCG at k. relevances sorted by predicted score descending."""
    dcg = _dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_ndcg_per_user(
    user_ids: list[int],
    scores: list[float],
    labels: list[float],
    k_values: list[int] = NDCG_K,
) -> dict[int, float]:
    """Compute NDCG@k per user. Returns mean NDCG for each k."""
    from collections import defaultdict
    by_user: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for uid, s, l in zip(user_ids, scores, labels):
        by_user[uid].append((s, l))

    ndcg_by_k = {k: [] for k in k_values}
    for uid, pairs in by_user.items():
        pairs_sorted = sorted(pairs, key=lambda x: -x[0])  # by score desc
        relevances = [p[1] for p in pairs_sorted]
        for k in k_values:
            ndcg_by_k[k].append(_ndcg_at_k(relevances, k))

    return {k: sum(v) / len(v) if v else 0.0 for k, v in ndcg_by_k.items()}


def compute_metrics(
    scores: list[float],
    labels: list[float],
    user_ids: list[int] | None = None,
) -> dict[str, float]:
    """Compute AUC, accuracy, precision, recall, F1, and optionally NDCG@k."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError:
        raise ImportError("scikit-learn required for evaluation. Run: pip install scikit-learn")

    preds = [1 if s >= 0 else 0 for s in scores]
    labels_arr = [int(l) for l in labels]

    metrics = {}
    if len(set(labels_arr)) > 1:
        metrics["auc"] = roc_auc_score(labels_arr, scores)
    else:
        metrics["auc"] = 0.0

    metrics["accuracy"] = accuracy_score(labels_arr, preds)
    metrics["precision"] = precision_score(labels_arr, preds, zero_division=0)
    metrics["recall"] = recall_score(labels_arr, preds, zero_division=0)
    metrics["f1"] = f1_score(labels_arr, preds, zero_division=0)

    if user_ids:
        ndcg = compute_ndcg_per_user(user_ids, scores, labels)
        for k, v in ndcg.items():
            metrics[f"ndcg@{k}"] = v

    return metrics


def evaluate(model, val_loader, device) -> dict[str, float]:
    """Run model on validation set and compute all metrics."""
    model.eval()
    all_scores = []
    all_labels = []
    all_user_ids = []
    with torch.no_grad():
        for batch in val_loader:
            u = batch["user_vec"].to(device)
            p = batch["plant_vec"].to(device)
            scores = model(u, p)
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(batch["label"].tolist())
            all_user_ids.extend(batch["user_id"].tolist())
    return compute_metrics(all_scores, all_labels, all_user_ids)


def _fetch_plant_embeddings_from_mongo(plant_ids: list[int]) -> dict[int, list[float]]:
    """Fetch plant_tower_embedding (64-dim) from MongoDB for each plant_id. Returns dict plant_id -> embedding."""
    try:
        from pymongo import MongoClient
    except ImportError:
        return {}

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    coll_name = os.getenv("NEW_PLANT_COLLECTION", "NewPlantCollection")

    if not mongo_uri or not plant_ids:
        return {}

    client = MongoClient(mongo_uri)
    coll = client[mongo_db][coll_name]
    cursor = coll.find(
        {"$or": [{"plant_id": {"$in": plant_ids}}, {"id": {"$in": plant_ids}}], "plant_tower_embedding": {"$exists": True}},
        {"plant_id": 1, "id": 1, "plant_tower_embedding": 1},
    )
    result = {}
    for doc in cursor:
        pid = doc.get("plant_id") or doc.get("id")
        emb = doc.get("plant_tower_embedding")
        if pid is not None and emb and len(emb) == 64:
            result[int(pid)] = [float(x) for x in emb]
    return result


def _compute_user_plant_aggregates(
    interactions: list[dict],
    plant_embeddings: dict[int, list[float]],
) -> dict[int, dict]:
    """
    Compute plants_grown_agg and plants_killed_agg per user from interactions.
    Returns dict[user_id, {plants_grown_agg: list[64], plants_killed_agg: list[64]}].
    Uses zeros (64) when no plants in that category.
    """
    from collections import defaultdict

    ZERO_64 = [0.0] * 64
    by_user_grown: dict[int, list[list[float]]] = defaultdict(list)
    by_user_killed: dict[int, list[list[float]]] = defaultdict(list)

    for r in interactions:
        uid, pid, label = r["user_id"], r["plant_id"], r.get("label", 0)
        emb = plant_embeddings.get(pid)
        if not emb:
            continue
        if label == 1:
            by_user_grown[uid].append(emb)
        else:
            by_user_killed[uid].append(emb)

    result = {}
    all_user_ids = set(by_user_grown.keys()) | set(by_user_killed.keys())
    for uid in all_user_ids:
        grown = by_user_grown.get(uid, [])
        killed = by_user_killed.get(uid, [])
        if grown:
            grown_agg = [sum(x[i] for x in grown) / len(grown) for i in range(64)]
        else:
            grown_agg = ZERO_64.copy()
        if killed:
            killed_agg = [sum(x[i] for x in killed) / len(killed) for i in range(64)]
        else:
            killed_agg = ZERO_64.copy()
        result[uid] = {"plants_grown_agg": grown_agg, "plants_killed_agg": killed_agg}
    return result


def _get_all_plant_ids_from_feast(repo_path: Path | None) -> list[int]:
    """Get all plant_ids that have features in Feast (from parquet)."""
    repo = repo_path or (ROOT / "feature_repo")
    parquet_path = repo / "data" / "plant_features.parquet"
    if not parquet_path.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path, columns=["plant_id"])
        return sorted(df["plant_id"].dropna().astype(int).unique().tolist())
    except Exception:
        return []


def push_plant_embeddings_to_mongo(
    model,
    plant_ids: list[int],
    device,
    repo_path: Path | None = None,
) -> int:
    """Encode plants from Feast and update MongoDB (NEW_PLANT_COLLECTION). Returns count updated."""
    try:
        from pymongo import MongoClient
    except ImportError:
        print("pymongo not installed. Skipping MongoDB update.")
        return 0

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    coll_name = os.getenv("NEW_PLANT_COLLECTION", "NewPlantCollection")

    if not mongo_uri:
        print("MONGO_URI not set. Skipping MongoDB update.")
        return 0

    if not plant_ids:
        print("No plant IDs to push.")
        return 0

    from resources.two_tower_training.two_tower_model import _fetch_batch_features_from_feast

    batch_size = 256
    all_embeddings = []
    for i in range(0, len(plant_ids), batch_size):
        batch_ids = plant_ids[i : i + batch_size]
        vecs = _fetch_batch_features_from_feast("plant", batch_ids, repo_path=repo_path)
        with torch.no_grad():
            emb = model.encode_plant(vecs.to(device))
        all_embeddings.append(emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    emb_by_id = {pid: embeddings[i].tolist() for i, pid in enumerate(plant_ids)}

    client = MongoClient(mongo_uri)
    coll = client[mongo_db][coll_name]
    updated = 0
    for pid, emb in emb_by_id.items():
        result = coll.update_one(
            {"$or": [{"plant_id": pid}, {"id": pid}]},
            {"$set": {"plant_tower_embedding": emb, "plant_id": pid}},
        )
        updated += result.modified_count
    return updated


def load_and_prepare_data(
    repo_path: str | Path | None = None,
    use_aggregates: bool = True,
):
    """Load interactions, fetch user/plant features from Feast. Optionally aggregate plant embeddings from MongoDB."""
    with open(INTERACTIONS_PATH) as f:
        interactions = json.load(f)

    user_ids = sorted({r["user_id"] for r in interactions})
    plant_ids = sorted({r["plant_id"] for r in interactions})

    from resources.two_tower_training.two_tower_model import load_features_from_feast
    user_by_id, plant_by_id = load_features_from_feast(user_ids, plant_ids, repo_path=repo_path)

    if use_aggregates:
        plant_embeddings = _fetch_plant_embeddings_from_mongo(plant_ids)
        n_with_emb = len(plant_embeddings)
        if n_with_emb < len(plant_ids):
            print(f"  Warning: {len(plant_ids) - n_with_emb} plants missing plant_tower_embedding in MongoDB (using zeros)")
        aggregates = _compute_user_plant_aggregates(interactions, plant_embeddings)
        for uid, user in user_by_id.items():
            agg = aggregates.get(uid, {"plants_grown_agg": [0.0] * 64, "plants_killed_agg": [0.0] * 64})
            user["plants_grown_agg"] = agg["plants_grown_agg"]
            user["plants_killed_agg"] = agg["plants_killed_agg"]
    else:
        for user in user_by_id.values():
            user["plants_grown_agg"] = [0.0] * 64
            user["plants_killed_agg"] = [0.0] * 64

    return interactions, user_by_id, plant_by_id


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feast-repo", default=None, help="Path to Feast feature repo")
    parser.add_argument("--no-update-mongo", action="store_true", help="Skip pushing plant embeddings to MongoDB")
    parser.add_argument("--no-upload", action="store_true", help="Skip pushing plant embeddings to MongoDB (alias for --no-update-mongo)")
    parser.add_argument("--no-aggregates", action="store_true", help="Skip plant aggregates (use 21-dim user when MongoDB has no plant_tower_embedding)")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data from Feast...")
    repo_path = Path(args.feast_repo) if args.feast_repo else None
    use_aggregates = not getattr(args, "no_aggregates", False)
    interactions, user_by_id, plant_by_id = load_and_prepare_data(
        repo_path=repo_path,
        use_aggregates=use_aggregates,
    )
    print(f"  {len(interactions)} interactions, {len(user_by_id)} users, {len(plant_by_id)} plants")

    # Train/val split
    random.shuffle(interactions)
    n_val = int(len(interactions) * VAL_RATIO)
    val_interactions = interactions[:n_val]
    train_interactions = interactions[n_val:]

    train_ds = InteractionDataset(train_interactions, user_by_id, plant_by_id)
    val_ds = InteractionDataset(val_interactions, user_by_id, plant_by_id)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    from resources.two_tower_training.two_tower_model import create_two_tower_model
    model = create_two_tower_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            u = batch["user_vec"].to(device)
            p = batch["plant_vec"].to(device)
            labels = batch["label"].to(device)

            scores = model(u, p)
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                u = batch["user_vec"].to(device)
                p = batch["plant_vec"].to(device)
                labels = batch["label"].to(device)
                scores = model(u, p)
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                val_loss += loss.item()
                n_val_batches += 1
        val_loss /= max(n_val_batches, 1)

        val_metrics = evaluate(model, val_loader, device)
        metric_str = "  ".join(
            f"{k}={v:.4f}" for k, v in sorted(val_metrics.items())
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )
        print(f"  Val: {metric_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = OUTPUT_DIR / "two_tower.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint to {ckpt_path}")

    # Final evaluation with best model
    ckpt = torch.load(
        OUTPUT_DIR / "two_tower.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ckpt["model_state"])
    final_metrics = evaluate(model, val_loader, device)
    print("\nFinal validation metrics:")
    for k, v in sorted(final_metrics.items()):
        print(f"  {k}: {v:.4f}")

    if not args.no_update_mongo and not getattr(args, "no_upload", False):
        plant_ids = _get_all_plant_ids_from_feast(repo_path)
        if not plant_ids:
            plant_ids = sorted(plant_by_id.keys())
        print(f"\nPushing plant embeddings to MongoDB ({len(plant_ids)} plants)...")
        n = push_plant_embeddings_to_mongo(model, plant_ids, device, repo_path)
        print(f"Updated {n} plants in MongoDB.")

    print("Training complete.")


if __name__ == "__main__":
    main()
