"""
Offline evaluation of the new TwoTowerModel (categorical_embedding + aggregate user dim).

Training (``training_script.py``) uses **BPR** on positive interactions with sampled negatives;
this eval measures **retrieval** quality (Recall@K, NDCG@K, Hit@K) — the same ranking objective
at evaluation time. User aggregates are still **mean plant-tower 64-d** over grown vs killed from
interactions, matching training.

Ground truth (default **holdout**): per-user 80/20 split of ``synthetic_interactions.json``;
relevant plants = test interactions with ``label == 1`` (no leakage). Aggregates use **train**
split rows only when aggregates are enabled.

Alternative (**oracle**): ``oracle_score`` on the **JSON** plant catalog (full plant docs); model
scoring still uses Feast plant features when ``--plants-source feast`` (default).

**Default plant/user 21-d features: Feast** (``plant_features`` / ``user_features``), aligned
with training. Use ``--plants-source json`` for offline eval without a materialized Feast store.

64-d aggregates: prefer ``plant_tower_features`` from Feast; missing IDs are filled by encoding
the plant tower from 21-d (same as training fallback).

Data:
  - Users: synthetic_users.json (merged with Feast 21-d when using Feast)
  - Interactions: synthetic_interactions.json
  - Plants (model): Feast parquets by default, or local JSON catalog
  - Model: ``output/two_tower.pt`` or ``two_tower.pt`` next to this file

Optional semantic baseline (``baseline_semantic.py``): cosine user vs plant ``profile_embedding``
in MongoDB over the full Mongo plant list. Use ``--no-baseline`` to skip.

Run from project root:
  python -m resources.two_tower_training.eval
  python -m resources.two_tower_training.eval --ground-truth oracle
  python -m resources.two_tower_training.eval --plants-source json
  python -m resources.two_tower_training.eval --feast-repo /path/to/feature_repo
  python -m resources.two_tower_training.eval --n-users 100 --seed 42
  python -m resources.two_tower_training.eval --no-aggregates
  python -m resources.two_tower_training.eval --no-baseline
  python -m resources.two_tower_training.eval --cohere-rerank

**Cohere rerank** (``--cohere-rerank``): optional track using ``COHERE_API_KEY`` and text from Mongo
(same idea as ``backend/recommend``). Tuning: ``--retrieve-k``, ``--rerank-top-m``.
Semantic **baseline** does not use Cohere.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resources.ETL.feature_engineer import apply_categorical_embeddings, apply_user_embeddings
from resources.synthetic_user.generate_interactions import MATCH_THRESHOLDS, oracle_score
from resources.two_tower_training.baseline_semantic import (
    baseline_semantic_rank,
    ensure_user_semantic_embeddings,
    load_semantic_plants_from_mongo,
)
from resources.two_tower_training.rerank_eval import prefetch_mongo_plants_for_rerank, two_tower_then_cohere_rerank
from resources.two_tower_training.two_tower_model import (
    TwoTowerModel,
    fetch_plant_tower_embeddings_from_feast,
    load_features_from_feast,
)

USER_PATH_CANDIDATES = [
    ROOT / "resources" / "data" / "synthetic_users.json",
    ROOT / "resources" / "synthetic_user" / "synthetic_users.json",
]
PLANTS_PATH = ROOT / "resources" / "data_creating" / "permapeople_plants_mapped_normalized.json"
PLANTS_ALT = ROOT / "resources" / "data_creating" / "permapeople_plants_mapped_uniform.json"
INTERACTIONS_PATH = ROOT / "resources" / "two_tower_training" / "synthetic_interactions.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MODEL_PATH = OUTPUT_DIR / "two_tower.pt"
MODEL_PATH_ALT = Path(__file__).resolve().parent / "two_tower.pt"

EVAL_KS = (5, 10, 20)
AGG_DIM = 64


def _norm(val: str | None) -> str | None:
    if val is None:
        return None
    return str(val).strip().lower() or None


def resolve_users_path() -> Path:
    for p in USER_PATH_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No synthetic_users.json found. Tried:\n  "
        + "\n  ".join(str(p) for p in USER_PATH_CANDIDATES)
    )


def load_plants() -> list[dict]:
    path = PLANTS_PATH if PLANTS_PATH.exists() else PLANTS_ALT
    with open(path) as f:
        data = json.load(f)
    return data.get("plants", data) if isinstance(data, dict) else data


def resolve_model_path() -> Path:
    """Prefer output/two_tower.pt; fall back to two_tower.pt beside eval (training default)."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    if MODEL_PATH_ALT.exists():
        return MODEL_PATH_ALT
    return MODEL_PATH


def plant_ids_from_feast_parquet(repo_path: Path | None) -> list[int]:
    """Plant entity ids from Feast offline store parquet (same as training_script)."""
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


def merge_user_categorical_from_feast(all_users: list[dict], user_by_id: dict[int, dict]) -> int:
    """Overwrite categorical_embedding from Feast. Returns count updated."""
    n = 0
    for u in all_users:
        uid = u.get("user_id")
        if uid is None:
            continue
        fe = user_by_id.get(int(uid))
        emb = fe.get("categorical_embedding") if fe else None
        if emb and len(emb) == 21:
            u["categorical_embedding"] = list(emb)
            n += 1
    return n


def plant_tower_64_for_aggregates(
    model: TwoTowerModel,
    plant_rows: list[tuple[int, list[float]]],
    device: torch.device,
    repo_path: Path | None,
    batch_size: int,
) -> dict[int, list[float]]:
    """
    64-d vectors for grown/killed aggregates: Feast plant_tower_features when present,
    else encode plant tower from 21-d (matches training_script when tower missing in Feast).
    """
    ids = [pid for pid, _ in plant_rows]
    from_feast = fetch_plant_tower_embeddings_from_feast(ids, repo_path=repo_path)
    missing = [pid for pid in ids if pid not in from_feast]
    if not missing:
        return from_feast
    id_to_emb = {pid: emb for pid, emb in plant_rows}
    rows = [(pid, id_to_emb[pid]) for pid in missing if pid in id_to_emb]
    if rows:
        from_feast.update(encode_all_plants_tower64(model, rows, device, batch_size=batch_size))
    return from_feast


def split_interactions_holdout(
    interactions: list[dict],
    rng: random.Random,
    train_frac: float = 0.8,
) -> tuple[list[dict], dict[int, set[int]], int, int]:
    """
    Per user with ≥2 interactions: shuffle, split into train / test.
    Users with <2 interactions: all rows stay in train (excluded from holdout positives).

    Returns:
        train_interactions (for aggregates),
        test_positives_by_uid (plant_id where label==1 in the test split),
        n_train_rows, n_test_rows.
    """
    by_uid: dict[int, list[dict]] = defaultdict(list)
    for r in interactions:
        uid = r.get("user_id")
        if uid is None:
            continue
        by_uid[int(uid)].append(r)

    train_list: list[dict] = []
    test_positives: dict[int, set[int]] = defaultdict(set)
    n_test_rows = 0

    for uid, rows in by_uid.items():
        if len(rows) < 2:
            train_list.extend(rows)
            continue
        rows_copy = list(rows)
        rng.shuffle(rows_copy)
        n = len(rows_copy)
        split = int(train_frac * n)
        if split <= 0:
            split = 1
        if split >= n:
            split = n - 1
        train_part = rows_copy[:split]
        test_part = rows_copy[split:]
        train_list.extend(train_part)
        n_test_rows += len(test_part)
        for r in test_part:
            if int(r.get("label", 0)) != 1:
                continue
            pid = r.get("plant_id")
            if pid is None:
                continue
            test_positives[uid].add(int(pid))

    return train_list, dict(test_positives), len(train_list), n_test_rows


def oracle_positives(user: dict, plants: list[dict]) -> set[int]:
    care = _norm(user.get("care_level")) or "medium"
    thr = MATCH_THRESHOLDS.get(care, 0.40)
    out = set()
    for plant in plants:
        pid = plant.get("id")
        if pid is None:
            continue
        if oracle_score(user, plant) >= thr:
            out.add(int(pid))
    return out


def user_model_vector(user: dict) -> list[float]:
    """149-d input: 21 cat + plants_grown_agg + plants_killed_agg (zeros if missing)."""
    cat = user.get("categorical_embedding")
    if not cat or len(cat) != 21:
        raise ValueError("User missing 21-d categorical_embedding; run apply_user_embeddings first.")
    grown = user.get("plants_grown_agg") or [0.0] * AGG_DIM
    killed = user.get("plants_killed_agg") or [0.0] * AGG_DIM
    grown = list(grown)[:AGG_DIM] + [0.0] * max(0, AGG_DIM - len(grown))
    killed = list(killed)[:AGG_DIM] + [0.0] * max(0, AGG_DIM - len(killed))
    return list(cat) + grown + killed


def _cohere_error_bucket(msg: str | None) -> str:
    """Rough grouping for Cohere API failure strings (printed in eval summary)."""
    if not msg:
        return "unknown"
    m = msg.lower()
    if msg.startswith("HTTP "):
        rest = msg[5:].split("|", 1)[0].strip()
        try:
            code = int(rest.split()[0])
            if code == 429:
                return "rate_limit"
            if code in (401, 403):
                return "auth"
            if code in (502, 503, 504):
                return "upstream_unavailable"
            if 400 <= code < 500:
                return "client_error"
        except (ValueError, IndexError):
            pass
    if "429" in msg or "rate" in m or "too many" in m:
        return "rate_limit"
    if "401" in msg or "403" in msg or "unauthorized" in m or "forbidden" in m:
        return "auth"
    if "timeout" in m or "timed out" in m:
        return "timeout"
    if "503" in msg or "502" in msg or "504" in msg or "overloaded" in m:
        return "upstream_unavailable"
    return "other"


def _cohere_msg_one_line(msg: str, max_len: int = 140) -> str:
    s = " ".join(msg.split())
    return s[:max_len] + ("..." if len(s) > max_len else "")


def compute_user_plant_aggregates(
    interactions: list[dict],
    plant_tower_64_by_id: dict[int, list[float]],
) -> dict[int, dict[str, list[float]]]:
    """
    Per user: mean of plant-tower 64-d vectors for label==1 (grown) and label==0 (killed).
    Matches resources/two_tower_training/training_script._compute_user_plant_aggregates.
    """
    zero_64 = [0.0] * AGG_DIM
    by_grown: dict[int, list[list[float]]] = defaultdict(list)
    by_killed: dict[int, list[list[float]]] = defaultdict(list)

    for r in interactions:
        uid = r.get("user_id")
        pid = r.get("plant_id")
        if uid is None or pid is None:
            continue
        emb = plant_tower_64_by_id.get(int(pid))
        if not emb:
            continue
        label = int(r.get("label", 0))
        if label == 1:
            by_grown[uid].append(emb)
        else:
            by_killed[uid].append(emb)

    result: dict[int, dict[str, list[float]]] = {}
    all_uids = set(by_grown.keys()) | set(by_killed.keys())
    for uid in all_uids:
        grown = by_grown.get(uid, [])
        killed = by_killed.get(uid, [])
        grown_agg = (
            [sum(x[i] for x in grown) / len(grown) for i in range(AGG_DIM)] if grown else zero_64.copy()
        )
        killed_agg = (
            [sum(x[i] for x in killed) / len(killed) for i in range(AGG_DIM)] if killed else zero_64.copy()
        )
        result[uid] = {"plants_grown_agg": grown_agg, "plants_killed_agg": killed_agg}
    return result


def attach_aggregates_to_users(users: list[dict], aggregates: dict[int, dict[str, list[float]]]) -> None:
    """Set plants_grown_agg / plants_killed_agg on each user (zeros if user not in aggregates)."""
    z = {"plants_grown_agg": [0.0] * AGG_DIM, "plants_killed_agg": [0.0] * AGG_DIM}
    for u in users:
        uid = u.get("user_id")
        if uid is None:
            continue
        agg = aggregates.get(uid, z)
        u["plants_grown_agg"] = list(agg["plants_grown_agg"])
        u["plants_killed_agg"] = list(agg["plants_killed_agg"])


@torch.no_grad()
def encode_all_plants_tower64(
    model: TwoTowerModel,
    plant_rows: list[tuple[int, list[float]]],
    device: torch.device,
    batch_size: int = 512,
) -> dict[int, list[float]]:
    """plant_id -> 64-d L2-normalized plant tower output (same space as training aggregates)."""
    out: dict[int, list[float]] = {}
    for i in range(0, len(plant_rows), batch_size):
        chunk = plant_rows[i : i + batch_size]
        pids = [c[0] for c in chunk]
        p_mat = torch.tensor([c[1] for c in chunk], dtype=torch.float32, device=device)
        e = model.encode_plant(p_mat)
        for j, pid in enumerate(pids):
            out[pid] = e[j].cpu().tolist()
    return out


def recall_at_k(relevant: set[int], ranked_ids: list[int], k: int) -> float:
    if not relevant:
        return 1.0
    top = set(ranked_ids[:k])
    return len(relevant & top) / len(relevant)


def ndcg_at_k(relevant: set[int], ranked_ids: list[int], k: int) -> float:
    if not relevant:
        return 1.0
    top_k = ranked_ids[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, pid in enumerate(top_k) if pid in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(relevant: set[int], ranked_ids: list[int], k: int) -> float:
    if not relevant:
        return 1.0
    return 1.0 if (relevant & set(ranked_ids[:k])) else 0.0


def _p95(latencies_ms: list[float]) -> float:
    if not latencies_ms:
        return 0.0
    s = sorted(latencies_ms)
    return s[max(0, int(len(s) * 0.95) - 1)]


@torch.no_grad()
def score_all_plants(
    model: TwoTowerModel,
    user_vec: list[float],
    plant_rows: list[tuple[int, list[float]]],
    device: torch.device,
    batch_size: int = 512,
) -> list[tuple[int, float]]:
    """Return [(plant_id, score), ...] sorted by score descending."""
    u_base = torch.tensor(user_vec, dtype=torch.float32, device=device)
    scores: list[tuple[int, float]] = []
    for i in range(0, len(plant_rows), batch_size):
        chunk = plant_rows[i : i + batch_size]
        pids = [c[0] for c in chunk]
        p_mat = torch.tensor([c[1] for c in chunk], dtype=torch.float32, device=device)
        b = p_mat.size(0)
        u_b = u_base.unsqueeze(0).expand(b, -1)
        s = model(u_b, p_mat)
        for j, pid in enumerate(pids):
            scores.append((pid, float(s[j].item())))
    scores.sort(key=lambda x: -x[1])
    return scores


def load_model(device: torch.device) -> TwoTowerModel:
    path = resolve_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path} (also tried {MODEL_PATH} and {MODEL_PATH_ALT})"
        )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt)
    model = TwoTowerModel()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-tower retrieval eval (new model + synthetic users)")
    parser.add_argument("--n-users", type=int, default=100, help="Number of users to sample (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for user sampling and holdout split")
    parser.add_argument("--batch-size", type=int, default=512, help="Plant batch size for scoring")
    parser.add_argument(
        "--ground-truth",
        choices=("holdout", "oracle"),
        default="holdout",
        help="holdout: train/test split of synthetic_interactions (test label==1 relevant); "
        "oracle: oracle_score threshold (legacy)",
    )
    parser.add_argument(
        "--holdout-train-frac",
        type=float,
        default=0.8,
        help="Train fraction per user for holdout split (default 0.8)",
    )
    parser.add_argument(
        "--no-aggregates",
        action="store_true",
        help="Use zeros for plants_grown_agg / plants_killed_agg (ignore synthetic_interactions.json)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip semantic baseline (cosine: user vs plant profile_embedding in MongoDB)",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write metrics JSON")
    parser.add_argument(
        "--feast-repo",
        default=None,
        help="Path to Feast feature repo (default: project feature_repo)",
    )
    parser.add_argument(
        "--plants-source",
        choices=("feast", "json"),
        default="feast",
        help="Plant (and user) 21-d features: feast (default, matches training) or local JSON",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=36,
        help="Max two-tower candidates to pool for Cohere rerank (default 36)",
    )
    parser.add_argument(
        "--rerank-top-m",
        type=int,
        default=20,
        help="Cohere rerank top_n / prefix size (default 20)",
    )
    parser.add_argument(
        "--cohere-rerank",
        action="store_true",
        help="Run Cohere rerank (COHERE_API_KEY; pip install cohere) on Mongo plant text",
    )
    parser.add_argument(
        "--cohere-inter-request-ms",
        type=int,
        default=0,
        help="Optional delay (ms) after each user's Cohere rerank to reduce 429 rate limits (e.g. 200–500)",
    )
    args = parser.parse_args()

    # Load .env before Mongo / Voyage (database.mongodb reads MONGO_URI at import time).
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    users_path = resolve_users_path()
    print(f"Users: {users_path}")
    with open(users_path) as f:
        all_users = json.load(f)

    apply_user_embeddings(all_users)

    repo_path = Path(args.feast_repo) if args.feast_repo else (ROOT / "feature_repo")
    plants_oracle: list[dict] | None = None

    if args.plants_source == "feast":
        plant_ids_feast = plant_ids_from_feast_parquet(repo_path)
        if not plant_ids_feast:
            print(
                f"ERROR: No plants in Feast ({repo_path / 'data' / 'plant_features.parquet'}). "
                "Run ETL (resources.ETL.flow) or use --plants-source json."
            )
            sys.exit(1)
        user_ids_sorted = sorted(
            int(u["user_id"]) for u in all_users if u.get("user_id") is not None
        )
        print(
            f"Feast: loading 21-d features for {len(user_ids_sorted)} users, {len(plant_ids_feast)} plants "
            f"({repo_path})"
        )
        user_by_feast, plant_by_feast = load_features_from_feast(
            user_ids_sorted, plant_ids_feast, repo_path=repo_path
        )
        n_ue = merge_user_categorical_from_feast(all_users, user_by_feast)
        print(f"  Merged Feast user categorical_embedding for {n_ue} users")

        plant_rows = []
        for pid in plant_ids_feast:
            p = plant_by_feast.get(pid)
            emb = p.get("categorical_embedding") if p else None
            if emb and len(emb) == 21:
                plant_rows.append((int(pid), list(emb)))
        print(f"Plants with Feast 21-d embeddings: {len(plant_rows)}")
        if args.ground_truth == "oracle":
            plants_oracle = load_plants()
    else:
        plants = load_plants()
        apply_categorical_embeddings(plants)
        plants_oracle = plants
        plant_rows = []
        for p in plants:
            pid = p.get("id")
            emb = p.get("categorical_embedding")
            if pid is None or not emb or len(emb) != 21:
                continue
            plant_rows.append((int(pid), emb))
        print(f"Plants with JSON embeddings: {len(plant_rows)} / {len(plants)}")

    plant_ids_set = {pid for pid, _ in plant_rows}

    model = load_model(device)
    model_path_used = resolve_model_path()

    use_agg = not args.no_aggregates
    test_positives_by_uid: dict[int, set[int]] = {}
    n_train_rows = n_test_rows = 0

    if args.ground_truth == "holdout":
        if not INTERACTIONS_PATH.exists():
            print(f"ERROR: --ground-truth holdout requires {INTERACTIONS_PATH}")
            sys.exit(1)
        with open(INTERACTIONS_PATH) as f:
            interactions_all = json.load(f)
        rng_split = random.Random(args.seed)
        train_interactions, test_positives_by_uid, n_train_rows, n_test_rows = split_interactions_holdout(
            interactions_all,
            rng_split,
            train_frac=args.holdout_train_frac,
        )
        n_hp = sum(1 for s in test_positives_by_uid.values() if s)
        print(
            f"Holdout split: {n_train_rows} train rows, {n_test_rows} test rows; "
            f"{n_hp} users with ≥1 test positive (label==1)."
        )
        print(f"Plant tower 64-d for aggregates (Feast tower + model fallback) — {len(plant_rows)} plants...")
        plant_64 = plant_tower_64_for_aggregates(
            model, plant_rows, device, repo_path, args.batch_size
        )
        if use_agg:
            aggregates = compute_user_plant_aggregates(train_interactions, plant_64)
            attach_aggregates_to_users(all_users, aggregates)
            print(
                f"Attached grown/killed aggregates from TRAIN split only ({len(train_interactions)} rows); "
                f"{len(aggregates)} users with labeled plants in embedding table."
            )
        else:
            attach_aggregates_to_users(all_users, {})
    else:
        # oracle ground truth: aggregates from all interactions if enabled
        if use_agg:
            if not INTERACTIONS_PATH.exists():
                print(f"WARNING: {INTERACTIONS_PATH} not found — using zero aggregates.")
                use_agg = False
            else:
                with open(INTERACTIONS_PATH) as f:
                    interactions = json.load(f)
                print(f"Plant tower 64-d for aggregates (Feast tower + model fallback) — {len(plant_rows)} plants...")
                plant_64 = plant_tower_64_for_aggregates(
                    model, plant_rows, device, repo_path, args.batch_size
                )
                aggregates = compute_user_plant_aggregates(interactions, plant_64)
                attach_aggregates_to_users(all_users, aggregates)
                print(
                    f"Attached grown/killed aggregates from {len(interactions)} interactions "
                    f"({len(aggregates)} users with at least one labeled plant in embedding table)."
                )
        if not use_agg:
            attach_aggregates_to_users(all_users, {})

    if args.ground_truth == "oracle" and not plants_oracle:
        print("ERROR: oracle ground truth requires JSON plant catalog (permapeople JSON).")
        sys.exit(1)

    uid_to_user = {int(u["user_id"]): u for u in all_users if u.get("user_id") is not None}

    if args.ground_truth == "holdout":
        eligible_uids = [
            uid
            for uid, pos in test_positives_by_uid.items()
            if (pos & plant_ids_set) and uid in uid_to_user
        ]
        if not eligible_uids:
            print("No users with ≥1 holdout test positive (label==1) in the scored plant catalog. Nothing to report.")
            sys.exit(1)
        random.seed(args.seed)
        n_take = min(args.n_users, len(eligible_uids))
        sampled_uids = random.sample(eligible_uids, n_take)
        sampled = [uid_to_user[uid] for uid in sampled_uids]
        print(
            f"Sampled users: {len(sampled)} (seed={args.seed}; {len(eligible_uids)} eligible with holdout positives in catalog)"
        )
    else:
        n_sample = min(args.n_users, len(all_users))
        random.seed(args.seed)
        sampled = random.sample(all_users, n_sample)
        print(f"Sampled users: {n_sample} (seed={args.seed})")

    max_k = max(EVAL_KS)
    from resources.two_tower_training.baseline_semantic import ensure_user_semantic_embeddings

    if not args.no_baseline:
        try:
            ensure_user_semantic_embeddings(sampled)
        except Exception as e:
            print(f"WARNING: ensure_user_semantic_embeddings failed ({e})")

    mongo_semantic: list[dict] | None = None
    mongo_ids: set[int] = set()
    use_baseline = not args.no_baseline
    if use_baseline:
        mongo_semantic, mongo_load_err = load_semantic_plants_from_mongo()
        if mongo_load_err is not None:
            print(
                f"WARNING: Could not read MongoDB for semantic baseline ({mongo_load_err}) — baseline skipped."
            )
            use_baseline = False
        elif not mongo_semantic:
            print(
                "WARNING: No plants with non-empty profile_embedding in MongoDB — semantic baseline skipped. "
                "Run plant ingest/upload so the plant catalog (NEW_PLANT_COLLECTION) includes profile_embedding, "
                "or use --no-baseline."
            )
            use_baseline = False
        else:
            mongo_ids = set()
            for d in mongo_semantic:
                pid = d.get("plant_id")
                if pid is None:
                    continue
                mongo_ids.add(int(pid) if not isinstance(pid, int) else pid)
            print(f"Baseline: MongoDB has {len(mongo_semantic)} plants with profile_embedding.")

    metrics = {k: {"recall": [], "ndcg": [], "hit": []} for k in EVAL_KS}
    cohere_metrics = {k: {"recall": [], "ndcg": [], "hit": []} for k in EVAL_KS}
    baseline_metrics = {k: {"recall": [], "ndcg": [], "hit": []} for k in EVAL_KS}
    baseline_latencies: list[float] = []
    two_tower_latencies_ms: list[float] = []
    cohere_latencies_ms: list[float] = []
    cohere_debug_rows: list[dict] = []
    skipped = 0
    use_cohere_rerank = bool(getattr(args, "cohere_rerank", False))

    mongo_rerank_cache: dict[int, dict] | None = None
    if use_cohere_rerank:
        catalog_pids = [pid for pid, _ in plant_rows]
        mongo_rerank_cache = prefetch_mongo_plants_for_rerank(catalog_pids)
        print(
            f"Mongo prefetch for Cohere: {len(mongo_rerank_cache)}/{len(catalog_pids)} plants "
            f"(one query, slim projection — no per-user round-trips)"
        )

    for u in sampled:
        try:
            uvec = user_model_vector(u)
        except ValueError:
            skipped += 1
            continue

        uid = int(u["user_id"]) if u.get("user_id") is not None else None
        if uid is None:
            skipped += 1
            continue

        if args.ground_truth == "holdout":
            rel = test_positives_by_uid.get(uid, set()) & plant_ids_set
        else:
            rel = oracle_positives(u, plants_oracle) & plant_ids_set
        if not rel:
            skipped += 1
            continue

        t_tt = time.perf_counter()
        ranked = score_all_plants(model, uvec, plant_rows, device, batch_size=args.batch_size)
        two_tower_latencies_ms.append((time.perf_counter() - t_tt) * 1000.0)
        order = [pid for pid, _ in ranked]

        for k in EVAL_KS:
            metrics[k]["recall"].append(recall_at_k(rel, order, k))
            metrics[k]["ndcg"].append(ndcg_at_k(rel, order, k))
            metrics[k]["hit"].append(hit_at_k(rel, order, k))

        if use_cohere_rerank:
            order_cohere, ch_dbg = two_tower_then_cohere_rerank(
                ranked,
                u,
                retrieve_k=args.retrieve_k,
                rerank_top_m=args.rerank_top_m,
                mongo_cache=mongo_rerank_cache,
            )
            # Cohere-only API time (co.rerank calls); excludes retry backoff sleep — see rerank_eval api_ms
            cohere_latencies_ms.append(float(ch_dbg.get("api_ms", 0.0)))
            cohere_debug_rows.append(ch_dbg)
            for k in EVAL_KS:
                cohere_metrics[k]["recall"].append(recall_at_k(rel, order_cohere, k))
                cohere_metrics[k]["ndcg"].append(ndcg_at_k(rel, order_cohere, k))
                cohere_metrics[k]["hit"].append(hit_at_k(rel, order_cohere, k))
            if args.cohere_inter_request_ms > 0:
                time.sleep(args.cohere_inter_request_ms / 1000.0)

        if use_baseline and mongo_semantic and u.get("profile_embedding"):
            if args.ground_truth == "holdout":
                rel_mongo = rel & mongo_ids
            else:
                rel_mongo = oracle_positives(u, plants_oracle) & mongo_ids
            if rel_mongo:
                t0 = time.perf_counter()
                baseline_pred = baseline_semantic_rank(u, mongo_semantic, k=max_k)
                baseline_latencies.append((time.perf_counter() - t0) * 1000)
                for k in EVAL_KS:
                    baseline_metrics[k]["recall"].append(recall_at_k(rel_mongo, baseline_pred, k))
                    baseline_metrics[k]["ndcg"].append(ndcg_at_k(rel_mongo, baseline_pred, k))
                    baseline_metrics[k]["hit"].append(hit_at_k(rel_mongo, baseline_pred, k))

    n_eval = len(metrics[EVAL_KS[0]]["recall"])
    if n_eval == 0:
        print("No users evaluated (no relevant items or invalid embeddings). Nothing to report.")
        sys.exit(1)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\nEvaluated: {n_eval} users (skipped {skipped})")
    print(f"Model: {model_path_used}")
    print(
        f"Ground truth: {args.ground_truth}"
        + (
            f" (train_frac={args.holdout_train_frac}; train_rows={n_train_rows}, test_rows={n_test_rows})"
            if args.ground_truth == "holdout"
            else ""
        )
    )
    print(f"Plants source: {args.plants_source}  (Feast repo: {repo_path})")
    print(f"User aggregates (grown/killed from interactions): {'yes' if use_agg else 'no (zeros)'}")
    print(f"Cohere rerank: {'on' if use_cohere_rerank else 'off'}")
    tt_lat_mean = mean(two_tower_latencies_ms)
    tt_lat_p95 = _p95(two_tower_latencies_ms)
    print("\nTwo-tower only (full catalog, Feast 21-d → model):")
    out: dict = {
        "ground_truth": args.ground_truth,
        "plants_source": args.plants_source,
        "feast_repo": str(repo_path),
        "n_users_evaluated": n_eval,
        "n_users_skipped": skipped,
        "n_plants": len(plant_rows),
        "use_aggregates": use_agg,
        "training_objective": "bpr_pairwise_ranking",
        "use_cohere_rerank": use_cohere_rerank,
        "latency_ms_mean": tt_lat_mean,
        "latency_ms_p95": tt_lat_p95,
        "ks": {},
    }
    if args.ground_truth == "holdout":
        out["holdout_train_frac"] = args.holdout_train_frac
        out["holdout_n_train_rows"] = n_train_rows
        out["holdout_n_test_rows"] = n_test_rows
    for k in EVAL_KS:
        r = mean(metrics[k]["recall"])
        n = mean(metrics[k]["ndcg"])
        h = mean(metrics[k]["hit"])
        print(f"  @{k}:  Recall={r:.4f}  NDCG={n:.4f}  Hit={h:.4f}")
        out["ks"][str(k)] = {"recall": r, "ndcg": n, "hit": h}
    print(
        f"  Latency (full catalog score_all_plants, per user): "
        f"mean={tt_lat_mean:.1f}ms  p95={tt_lat_p95:.1f}ms"
    )

    out["cohere_rerank"] = None
    if use_cohere_rerank and cohere_debug_rows:
        ch_mean = mean(cohere_latencies_ms) if cohere_latencies_ms else 0.0
        ch_p95 = _p95(cohere_latencies_ms) if cohere_latencies_ms else 0.0
        ch_ok = sum(1 for d in cohere_debug_rows if d.get("cohere") == "ok")
        ch_counts = dict(Counter(d.get("cohere") or "unknown" for d in cohere_debug_rows))
        err_buckets = Counter(
            _cohere_error_bucket(d.get("message"))
            for d in cohere_debug_rows
            if d.get("cohere") == "error"
        )
        err_samples = [
            _cohere_msg_one_line(d["message"])
            for d in cohere_debug_rows
            if d.get("cohere") == "error" and d.get("message")
        ][:3]
        print(
            f"\nTwo-tower + Cohere rerank ({args.rerank_top_m} docs, model rerank-v3.5):"
        )
        print(f"  Cohere stage ok: {ch_ok}/{len(cohere_debug_rows)} users")
        print(f"  Outcomes (why not ok): {ch_counts}")
        if err_buckets:
            print(f"  Error breakdown (from message text): {dict(err_buckets)}")
        if err_samples:
            print(f"  Sample errors (short, up to 3): {err_samples}")
        for k in EVAL_KS:
            r = mean(cohere_metrics[k]["recall"])
            nd = mean(cohere_metrics[k]["ndcg"])
            h = mean(cohere_metrics[k]["hit"])
            print(f"  @{k}:  Recall={r:.4f}  NDCG={nd:.4f}  Hit={h:.4f}")
        out["cohere_rerank"] = {
            "retrieve_k": args.retrieve_k,
            "rerank_top_m": args.rerank_top_m,
            "latency_ms_mean": ch_mean,
            "latency_ms_p95": ch_p95,
            "latency_note": "cohere.rerank() wall time only; excludes time.sleep() between retries",
            "users_cohere_ok": ch_ok,
            "cohere_outcome_counts": ch_counts,
            "cohere_error_buckets": dict(err_buckets) if err_buckets else {},
            "cohere_inter_request_ms": args.cohere_inter_request_ms,
            "ks": {},
        }
        for k in EVAL_KS:
            out["cohere_rerank"]["ks"][str(k)] = {
                "recall": mean(cohere_metrics[k]["recall"]),
                "ndcg": mean(cohere_metrics[k]["ndcg"]),
                "hit": mean(cohere_metrics[k]["hit"]),
            }
        print(
            f"  Latency (Cohere rerank API time only, per user; excludes retry sleep): "
            f"mean={ch_mean:.1f}ms  p95={ch_p95:.1f}ms"
        )

    n_bl = len(baseline_metrics[EVAL_KS[0]]["recall"])
    out["baseline"] = None
    if use_baseline and n_bl > 0:
        bl_mean = mean(baseline_latencies) if baseline_latencies else 0.0
        bl_p95 = _p95(baseline_latencies)
        out["baseline"] = {
            "n_users_evaluated": n_bl,
            "n_plants_mongo_semantic": len(mongo_semantic) if mongo_semantic else 0,
            "latency_ms_mean": bl_mean,
            "latency_ms_p95": bl_p95,
            "ks": {},
        }
        print(
            f"\nBaseline (cosine: user vs plant profile_embedding in Mongo; "
            f"mean over {n_bl} users with ground-truth positives in Mongo):"
        )
        for k in EVAL_KS:
            r = mean(baseline_metrics[k]["recall"])
            n = mean(baseline_metrics[k]["ndcg"])
            h = mean(baseline_metrics[k]["hit"])
            print(f"  @{k}:  Recall={r:.4f}  NDCG={n:.4f}  Hit={h:.4f}")
            out["baseline"]["ks"][str(k)] = {"recall": r, "ndcg": n, "hit": h}
        print(
            f"  Latency (baseline_semantic_rank per user): "
            f"mean={bl_mean:.1f}ms  p95={bl_p95:.1f}ms"
        )
    elif not args.no_baseline:
        print("\nBaseline: (not reported — Mongo unavailable, Voyage/embeddings failed, or no overlapping users)")

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
