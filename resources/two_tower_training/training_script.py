"""
Train two-tower model on synthetic interactions.
Loads features from Feast (no feature engineering). Trains with BPR loss
(positive interactions + hard negatives: sample S=64 uniform candidates, score, pick hardest).
Evaluates with AUC, accuracy, precision, recall, F1, NDCG@k.
After training, pushes plant tower embeddings to Feast by default
(``plant_tower_features`` view: 64-d ``tower_*`` fields). MongoDB embedding push is
currently disabled in this script (see post-train block).

MLflow (optional): logs params, BPR experiment notes (artifact), tags, per-epoch metrics,
final metrics, and ``two_tower.pt``.
Set ``MLFLOW_TRACKING_URI`` (default: local ``./mlruns``). Use ``--no-mlflow`` to disable.
Run from project root: python -m resources.two_tower_training.training_script
"""
import contextlib
import json
import math
import os
import random
import subprocess
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

from resources.synthetic_user.negative_sampling import (
    HARD_NEGATIVE_POOL_SIZE,
    TOP_H_HARD_NEGATIVES,
    pick_top_h_hard_negative_plant_embeddings,
    sample_uniform_negative_ids,
)

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INTERACTIONS_PATH = ROOT / "resources" / "two_tower_training" / "synthetic_interactions.json"
OUTPUT_DIR = Path(__file__).resolve().parent

BATCH_SIZE = 256
EPOCHS = 15
LR = 1e-3
VAL_RATIO = 0.1
SEED = 42
NDCG_K = [5, 10, 20]

# Logged to MLflow as artifact ``experiment_notes.txt`` for the BPR two-tower run.
BPR_EXPERIMENT_NOTES = """\
Two-tower recommendation — BPR (Bayesian Personalized Ranking) experiment

Objective
  Pairwise ranking: maximize score(user, positive_plant) vs score(user, negative_plant).
  Loss per pair: softplus(s_neg - s_pos) = -log sigmoid(s_pos - s_neg).

Training data
  Only interactions with label==1 (grown/success). Each step uses one positive (user, plant)
  and top_h=5 hard negatives: sample S=64 candidate plants uniformly from the eligible pool
  (excluding the positive and the user's other positives), score with the current model,
  take the top 5 scores, then BPR loss averaged over those five pairs.

Validation
  - val_loss: same BPR loss on val positives + sampled negatives.
  - AUC, accuracy, precision, recall, F1, NDCG@k: computed on the full val split (all labels),
    pointwise scores — same as before BPR.

Checkpoint
  Best val BPR loss when val has positives; else best train BPR loss.

User tower
  149-d = 21-d Feast categorical_embedding + 64-d plants_grown_agg + 64-d plants_killed_agg
  (aggregates from interaction history; negatives in BPR are not extra label rows in training).
"""


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


def _user_positives_from_interactions(interactions: list[dict]) -> dict[int, set[int]]:
    """user_id -> plant_ids with label==1 (grown)."""
    from collections import defaultdict

    out: dict[int, set[int]] = defaultdict(set)
    for r in interactions:
        if int(r.get("label", 0)) == 1:
            out[r["user_id"]].add(r["plant_id"])
    return dict(out)


class BPRInteractionDataset(Dataset):
    """
    One row per positive (user, plant) interaction.
    Each __getitem__ builds ``S`` uniform negative candidates (21-d plant embeddings) for hard mining.
    """

    def __init__(
        self,
        interactions_pos: list[dict],
        user_by_id: dict,
        plant_by_id: dict,
        all_plant_ids: list[int],
        user_positives: dict[int, set[int]],
        *,
        hard_negative_pool_size: int = HARD_NEGATIVE_POOL_SIZE,
        seed: int | None = None,
    ):
        self.plant_by_id = plant_by_id
        self.all_plant_ids = list(all_plant_ids)
        self.user_positives = user_positives
        self.s_neg = hard_negative_pool_size
        self._rng = random.Random(seed if seed is not None else SEED)
        self.samples: list[dict] = []
        for r in interactions_pos:
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
                    "plant_vec_pos": p_emb,
                    "user_id": uid,
                    "plant_id_pos": pid,
                })

    def __len__(self):
        return len(self.samples)

    def _eligible_negatives(self, uid: int, pid_pos: int) -> list[int]:
        pos_set = self.user_positives.get(uid, set())
        pool: list[int] = []
        for p in self.all_plant_ids:
            if p == pid_pos or p in pos_set:
                continue
            pl = self.plant_by_id.get(p)
            emb = pl.get("categorical_embedding") if pl else None
            if emb and len(emb) == 21:
                pool.append(p)
        if not pool:
            for p in self.all_plant_ids:
                if p != pid_pos:
                    pl = self.plant_by_id.get(p)
                    emb = pl.get("categorical_embedding") if pl else None
                    if emb and len(emb) == 21:
                        pool.append(p)
        return pool

    def __getitem__(self, idx):
        s = self.samples[idx]
        uid = s["user_id"]
        pid_pos = s["plant_id_pos"]
        neg_pool = self._eligible_negatives(uid, pid_pos)
        pid_neg_ids = sample_uniform_negative_ids(neg_pool, self.s_neg, self._rng)
        pos_emb = s["plant_vec_pos"]
        rows_list: list[torch.Tensor] = []
        for pid in pid_neg_ids:
            pl = self.plant_by_id.get(pid)
            emb = pl.get("categorical_embedding") if pl else None
            if emb and len(emb) == 21:
                rows_list.append(torch.tensor(emb, dtype=torch.float32))
            else:
                rows_list.append(torch.tensor(pos_emb, dtype=torch.float32))
        if not rows_list:
            plant_neg_cands = torch.tensor(pos_emb, dtype=torch.float32).unsqueeze(0).repeat(self.s_neg, 1)
        else:
            plant_neg_cands = torch.stack(rows_list, dim=0)
            if plant_neg_cands.shape[0] < self.s_neg:
                pad = plant_neg_cands[-1:].expand(self.s_neg - plant_neg_cands.shape[0], -1)
                plant_neg_cands = torch.cat([plant_neg_cands, pad], dim=0)
            elif plant_neg_cands.shape[0] > self.s_neg:
                plant_neg_cands = plant_neg_cands[: self.s_neg]
        return {
            "user_vec": torch.tensor(s["user_vec"], dtype=torch.float32),
            "plant_vec_pos": torch.tensor(s["plant_vec_pos"], dtype=torch.float32),
            "plant_neg_cands": plant_neg_cands,
            "user_id": uid,
            "plant_id_pos": pid_pos,
        }


def bpr_loss(scores_pos: torch.Tensor, scores_neg: torch.Tensor) -> torch.Tensor:
    """BPR: -log(sigmoid(s_pos - s_neg)) = softplus(s_neg - s_pos)."""
    return F.softplus(scores_neg - scores_pos).mean()


def bpr_loss_top_h(scores_pos: torch.Tensor, scores_neg: torch.Tensor) -> torch.Tensor:
    """BPR averaged over multiple negatives: ``scores_pos`` (B,), ``scores_neg`` (B, H)."""
    return F.softplus(scores_neg - scores_pos.unsqueeze(1)).mean()


def _mlflow_metric_key(name: str) -> str:
    """MLflow rejects @ in metric names (e.g. ndcg@5); map to ndcg_at_5."""
    return name.replace("@", "_at_")


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


def compute_plant_tower_embeddings(
    model,
    plant_ids: list[int],
    device,
    repo_path: Path | None = None,
) -> dict[int, list[float]]:
    """Encode plants from Feast 21-d inputs through the plant tower. Returns plant_id -> 64-d list."""
    if not plant_ids:
        return {}

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
    return {pid: embeddings[j].tolist() for j, pid in enumerate(plant_ids)}


def push_plant_embeddings_to_mongo(emb_by_id: dict[int, list[float]]) -> int:
    """Update MongoDB plant_tower_embedding (NEW_PLANT_COLLECTION). Returns count updated."""
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

    if not emb_by_id:
        print("No embeddings to push.")
        return 0

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
    """Load interactions, fetch user/plant 21-d features from Feast; aggregates use 64-d plant_tower from Feast."""
    with open(INTERACTIONS_PATH) as f:
        interactions = json.load(f)

    user_ids = sorted({r["user_id"] for r in interactions})
    plant_ids = sorted({r["plant_id"] for r in interactions})

    from resources.two_tower_training.two_tower_model import (
        fetch_plant_tower_embeddings_from_feast,
        load_features_from_feast,
    )
    user_by_id, plant_by_id = load_features_from_feast(user_ids, plant_ids, repo_path=repo_path)

    if use_aggregates:
        plant_embeddings = fetch_plant_tower_embeddings_from_feast(plant_ids, repo_path=repo_path)
        n_with_emb = len(plant_embeddings)
        if n_with_emb < len(plant_ids):
            print(
                f"  Warning: {len(plant_ids) - n_with_emb} plants missing plant_tower_features in Feast "
                f"(run a prior train push or ETL); those rows skipped in grown/killed aggregates"
            )
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


def _git_short_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


@contextlib.contextmanager
def _mlflow_tracking(enabled: bool, experiment_name: str):
    """
    Yields an ``mlflow`` module if tracking is enabled and installed, else ``None``.
    Respects env ``MLFLOW_EXPERIMENT_NAME`` over ``experiment_name`` when set.
    """
    if not enabled:
        yield None
        return
    try:
        import mlflow
    except ImportError:
        print("WARNING: mlflow not installed (pip install mlflow). Skipping MLflow tracking.")
        yield None
        return
    exp = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)
    mlflow.set_experiment(exp)
    with mlflow.start_run():
        yield mlflow


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feast-repo", default=None, help="Path to Feast feature repo")
    parser.add_argument("--no-update-mongo", action="store_true", help="Skip pushing plant embeddings to MongoDB")
    parser.add_argument("--no-upload", action="store_true", help="Skip pushing plant embeddings to MongoDB (alias for --no-update-mongo)")
    parser.add_argument(
        "--no-update-feast",
        action="store_true",
        help="Skip writing 64-d plant tower vectors to Feast (plant_tower_features.parquet)",
    )
    parser.add_argument(
        "--no-aggregates",
        action="store_true",
        help="Skip plant aggregates (zeros for grown/killed) when Feast has no plant_tower_features yet",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow experiment tracking")
    parser.add_argument(
        "--mlflow-experiment",
        default="two-tower",
        help="MLflow experiment name (default: two-tower; env MLFLOW_EXPERIMENT_NAME overrides)",
    )
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

    train_pos = [r for r in train_interactions if int(r.get("label", 0)) == 1]
    val_pos = [r for r in val_interactions if int(r.get("label", 0)) == 1]
    user_pos_train = _user_positives_from_interactions(train_interactions)
    user_pos_val = _user_positives_from_interactions(val_interactions)
    all_plant_ids = sorted(plant_by_id.keys())

    train_ds = BPRInteractionDataset(
        train_pos,
        user_by_id,
        plant_by_id,
        all_plant_ids,
        user_pos_train,
        hard_negative_pool_size=HARD_NEGATIVE_POOL_SIZE,
        seed=SEED,
    )
    val_ds = InteractionDataset(val_interactions, user_by_id, plant_by_id)
    val_bpr_ds = BPRInteractionDataset(
        val_pos,
        user_by_id,
        plant_by_id,
        all_plant_ids,
        user_pos_val,
        hard_negative_pool_size=HARD_NEGATIVE_POOL_SIZE,
        seed=SEED + 1,
    )
    if len(train_ds) == 0:
        raise RuntimeError(
            "BPR training needs at least one positive interaction in the train split; "
            "increase data or lower VAL_RATIO."
        )
    print(
        f"  Train BPR: {len(train_ds)} positives, Val: {len(val_ds)} rows "
        f"({len(val_bpr_ds)} val positives for val loss)"
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_bpr_loader = DataLoader(val_bpr_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    use_val_bpr_for_ckpt = len(val_bpr_ds) > 0
    if not use_val_bpr_for_ckpt:
        print("  WARNING: no positive labels in val split; checkpoint uses train BPR loss.")

    use_mlflow = not args.no_mlflow
    with _mlflow_tracking(use_mlflow, args.mlflow_experiment) as mlf:
        if mlf:
            mlf.log_params(
                {
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "lr": LR,
                    "val_ratio": VAL_RATIO,
                    "seed": SEED,
                    "ndcg_k": ",".join(str(k) for k in NDCG_K),
                    "use_aggregates": use_aggregates,
                    "no_update_mongo": args.no_update_mongo or getattr(args, "no_upload", False),
                    "no_update_feast": args.no_update_feast,
                    "n_interactions": len(interactions),
                    "n_train_samples": len(train_ds),
                    "n_val_samples": len(val_ds),
                    "n_users": len(user_by_id),
                    "n_plants": len(plant_by_id),
                    "loss": "bpr",
                    "bpr_train_positive_count": len(train_ds),
                    "bpr_val_positive_count": len(val_bpr_ds),
                    "bpr_checkpoint_on": "val_bpr_loss" if use_val_bpr_for_ckpt else "train_bpr_loss",
                    "bpr_train_pairs": "label_eq_1_only",
                    "bpr_negative_sampling": f"hard_sample_score_top{TOP_H_HARD_NEGATIVES}_of_s{HARD_NEGATIVE_POOL_SIZE}",
                    "bpr_hard_negative_pool_size": HARD_NEGATIVE_POOL_SIZE,
                    "bpr_top_h_hard_negatives": TOP_H_HARD_NEGATIVES,
                }
            )
            mlf.set_tag("device", str(device))
            mlf.set_tag("experiment", "two_tower_bpr")
            mlf.set_tag(
                "bpr_summary",
                "BPR pairwise ranking; train only positives; neg sampled; "
                "val metrics on full val rows; see artifact experiment_notes.txt",
            )
            gh = _git_short_hash()
            if gh:
                mlf.set_tag("git_commit", gh)
            try:
                mlf.log_text(BPR_EXPERIMENT_NOTES, "experiment_notes.txt")
            except Exception as e:
                print(f"WARNING: MLflow log_text experiment_notes failed: {e}")

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
                p_pos = batch["plant_vec_pos"].to(device)
                cand = batch["plant_neg_cands"].to(device)
                model.eval()
                p_neg_h = pick_top_h_hard_negative_plant_embeddings(
                    model, u, cand, top_h=TOP_H_HARD_NEGATIVES
                )
                model.train()
                bh, h, _ = p_neg_h.shape
                s_pos = model(u, p_pos)
                u_h = u.unsqueeze(1).expand(bh, h, -1).reshape(bh * h, -1)
                p_flat = p_neg_h.reshape(bh * h, -1)
                s_neg = model(u_h, p_flat).reshape(bh, h)
                loss = bpr_loss_top_h(s_pos, s_neg)

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
                if len(val_bpr_ds) > 0:
                    for batch in val_bpr_loader:
                        u = batch["user_vec"].to(device)
                        p_pos = batch["plant_vec_pos"].to(device)
                        cand = batch["plant_neg_cands"].to(device)
                        p_neg_h = pick_top_h_hard_negative_plant_embeddings(
                            model, u, cand, top_h=TOP_H_HARD_NEGATIVES
                        )
                        bh, h, _ = p_neg_h.shape
                        s_pos = model(u, p_pos)
                        u_h = u.unsqueeze(1).expand(bh, h, -1).reshape(bh * h, -1)
                        p_flat = p_neg_h.reshape(bh * h, -1)
                        s_neg = model(u_h, p_flat).reshape(bh, h)
                        loss = bpr_loss_top_h(s_pos, s_neg)
                        val_loss += loss.item()
                        n_val_batches += 1
                    val_loss /= max(n_val_batches, 1)
                else:
                    val_loss = float("nan")

            val_metrics = evaluate(model, val_loader, device)
            metric_str = "  ".join(
                f"{k}={v:.4f}" for k, v in sorted(val_metrics.items())
            )
            val_loss_str = f"{val_loss:.4f}" if math.isfinite(val_loss) else "nan"
            print(
                f"Epoch {epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss_str}"
            )
            print(f"  Val: {metric_str}")

            if mlf:
                log_me = {"train_loss": train_loss}
                if math.isfinite(val_loss):
                    log_me["val_loss"] = val_loss
                for mk, mv in val_metrics.items():
                    log_me[_mlflow_metric_key(f"val_{mk}")] = float(mv)
                mlf.log_metrics(log_me, step=epoch)

            if use_val_bpr_for_ckpt:
                improved = math.isfinite(val_loss) and val_loss < best_val_loss
                ckpt_metric = val_loss
            else:
                improved = train_loss < best_val_loss
                ckpt_metric = train_loss
            if improved:
                best_val_loss = ckpt_metric
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

        if mlf:
            for k, v in final_metrics.items():
                mlf.log_metric(_mlflow_metric_key(f"final_{k}"), float(v))
            mlf.log_metric("best_val_loss", float(best_val_loss))
            ckpt_path = OUTPUT_DIR / "two_tower.pt"
            if ckpt_path.is_file():
                mlf.log_artifact(str(ckpt_path))

        skip_mongo = args.no_update_mongo or getattr(args, "no_upload", False)
        if not skip_mongo or not args.no_update_feast:
            plant_ids = _get_all_plant_ids_from_feast(repo_path)
            if not plant_ids:
                plant_ids = sorted(plant_by_id.keys())
            print(f"\nComputing plant tower embeddings ({len(plant_ids)} plants)...")
            emb_by_id = compute_plant_tower_embeddings(model, plant_ids, device, repo_path)
            n_mongo = 0
            # MongoDB push disabled for now — uncomment when ready to sync plant_tower_embedding.
            # if not skip_mongo:
            #     print("Pushing plant embeddings to MongoDB...")
            #     n_mongo = push_plant_embeddings_to_mongo(emb_by_id)
            #     print(f"Updated {n_mongo} plants in MongoDB.")
            print("Skipping MongoDB plant embedding push (disabled in training_script).")
            if mlf:
                mlf.log_metric("mongo_plants_updated", float(n_mongo))
                mlf.set_tag("mongo_push", "disabled_in_script")
                mlf.log_metric("n_plant_embeddings_pushed", float(len(emb_by_id)))
            if not args.no_update_feast:
                print("Pushing plant tower embeddings to Feast (plant_tower_features)...")
                try:
                    from resources.ETL.feast_store import push_plant_tower_embeddings_to_feast

                    push_plant_tower_embeddings_to_feast(emb_by_id, repo_path)
                    print("Feast plant_tower_features updated.")
                    if mlf:
                        mlf.set_tag("feast_plant_tower_push", "ok")
                except Exception as e:
                    print(f"WARNING: Feast plant tower push failed: {e}")
                    if mlf:
                        mlf.set_tag("feast_plant_tower_push", "failed")

    print("Training complete.")


if __name__ == "__main__":
    main()
