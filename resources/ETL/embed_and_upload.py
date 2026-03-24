"""
Load mapped plant JSON → apply mappings → embed descriptions → upload to MongoDB.

Default input: resources/data_creating/permapeople_plants_mapped_uniform.json

Run from project root:
  python -m resources.ETL.embed_and_upload
  python -m resources.ETL.embed_and_upload --input path/to/other.json
  python -m resources.ETL.embed_and_upload --no-upload
  python -m resources.ETL.embed_and_upload --skip-embed
"""
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCES = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT_PATH = RESOURCES / "data_creating" / "permapeople_plants_mapped_uniform.json"
OUTPUT_PATH = RESOURCES / "data_creating" / "permapeople_plants_mapped_normalized.json"
SCHEMA_PATH = RESOURCES / "schema" / "permapeople_plants_sample.schema.json"
EMBED_MODEL = "voyage-4-lite"
EXCLUDED_KEYS = {"Danish name", "Dutch name", "German name"}
BATCH_SIZE = 64  # Smaller batches to reduce connection reset risk
MAX_RETRIES = 4
RETRY_DELAY = 2  # seconds, doubles each retry


def _collect_non_null(obj, prefix: str = "") -> list[str]:
    """Recursively collect non-null values as 'key: value' strings."""
    parts = []
    if obj is None:
        return parts
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            parts.extend(_collect_non_null(v, key))
    elif isinstance(obj, list):
        vals = [str(x) for x in obj if x is not None and str(x).strip()]
        if vals:
            parts.append(f"{prefix}: {', '.join(vals)}" if prefix else ", ".join(vals))
    elif isinstance(obj, (str, int, float, bool)):
        s = str(obj).strip()
        if s:
            parts.append(f"{prefix}: {s}" if prefix else s)
    return parts


def _format_ideal_tolerated(ideal: str | None, tolerated: str | None, label: str) -> str | None:
    """Format ideal/tolerated pair like mapping.py 2-subfield. Returns None if both empty."""
    if ideal and tolerated:
        return f"{label}: ideal {ideal}, tolerated {tolerated}"
    if ideal:
        return f"{label}: ideal {ideal}"
    if tolerated:
        return f"{label}: tolerated {tolerated}"
    return None


def plant_to_profile_text(plant: dict) -> str:
    """
    Concatenate all non-null fields into a single string for embedding.
    For light, water, soil (2 subfields like mapping.py): formats as "ideal X, tolerated Y".
    """
    parts = []

    # 2-subfield features (lighting, water, soil: ideal + tolerated)
    ec = plant.get("environment_care") or {}
    lr = ec.get("lighting") or {}
    wr = ec.get("water") or {}
    sr = ec.get("soil") or {}
    light_s = _format_ideal_tolerated(lr.get("ideal_light"), lr.get("tolerated_light"), "light")
    water_s = _format_ideal_tolerated(wr.get("ideal_water"), wr.get("tolerated_water"), "water")
    soil_s = _format_ideal_tolerated(sr.get("ideal_soil"), sr.get("tolerated_soil"), "soil")
    for s in (light_s, water_s, soil_s):
        if s:
            parts.append(s)

    # Rest: collect all other fields, excluding lighting/water/soil (already formatted)
    ec_copy = {k: v for k, v in ec.items() if k not in ("lighting", "water", "soil")}
    plant_copy = {**plant, "environment_care": ec_copy}
    parts.extend(_collect_non_null(plant_copy))

    return " | ".join(parts) if parts else "no profile"


def _load_schema_keys() -> tuple[list[str], list[str], list[str]]:
    """Load environment, info, problems keys from schema."""
    if not SCHEMA_PATH.exists():
        return [], [], []
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    env = [k for k in schema.get("environment", []) if k not in EXCLUDED_KEYS]
    info = [k for k in schema.get("info", []) if k not in EXCLUDED_KEYS]
    probs = schema.get("problems", [])
    return env, info, probs


def _to_uniform(plant: dict, env_keys: list[str], info_keys: list[str], prob_keys: list[str]) -> dict:
    """Transform plant with raw `data` array to uniform schema (environment_care, info, problems)."""
    data_dict = {}
    for item in plant.get("data", []):
        key = item.get("key")
        value = item.get("value")
        if key and key not in EXCLUDED_KEYS:
            data_dict[key] = value

    def fill_keys(keys: list[str], source: dict) -> dict:
        return {k: source.get(k) if source.get(k) else None for k in keys}

    environment_care = fill_keys(env_keys, data_dict)
    info = fill_keys(info_keys, data_dict)
    problems = fill_keys(prob_keys, data_dict)

    return {
        "id": plant.get("id"),
        "name": plant.get("name"),
        "slug": plant.get("slug"),
        "scientific_name": plant.get("scientific_name") or None,
        "description": plant.get("description") or None,
        "environment_care": environment_care,
        "info": info,
        "problems": problems,
        "created_at": plant.get("created_at"),
        "updated_at": plant.get("updated_at"),
        "type": plant.get("type"),
        "link": plant.get("link"),
        "images": plant.get("images"),
    }


def _needs_uniform_transform(plants: list[dict]) -> bool:
    """True if plants have raw `data` array format instead of environment_care."""
    if not plants:
        return False
    p = plants[0]
    if "data" in p and isinstance(p.get("data"), list) and p["data"]:
        ec = p.get("environment_care") or {}
        return not ec or "Light requirement" not in ec
    return False


def embed_plants(plants: list[dict]) -> None:
    """Add profile_embedding to each plant using Voyage AI. Retries on connection errors."""
    try:
        import voyageai
    except ImportError:
        print("voyageai not installed. Run: pip install voyageai")
        return
    texts = [plant_to_profile_text(p) for p in plants]
    print("\n--- Example of text being embedded (first plant) ---")
    print(texts[0][:1500] + ("..." if len(texts[0]) > 1500 else ""))
    print("---\n")
    print("Embedding with Voyage AI...")
    vo = voyageai.Client()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        delay = RETRY_DELAY
        for attempt in range(MAX_RETRIES):
            try:
                result = vo.embed(batch, model=EMBED_MODEL, input_type="document")
                for j, emb in enumerate(result.embeddings):
                    plants[i + j]["profile_embedding"] = emb
                print(f"  embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
                time.sleep(0.5)  # Brief pause between batches to reduce connection pressure
                break
            except (ConnectionError, OSError) as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  Connection error (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise


def main():
    input_path = INPUT_PATH
    if "--input" in sys.argv:
        i = sys.argv.index("--input")
        if i + 1 < len(sys.argv):
            input_path = Path(sys.argv[i + 1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return 1

    with open(input_path) as f:
        data = json.load(f)
    plants = data.get("plants", [])
    if not plants:
        print("No plants to load.")
        return 1

    print(f"Loaded {len(plants)} plants from {input_path}")

    if _needs_uniform_transform(plants):
        env_keys, info_keys, prob_keys = _load_schema_keys()
        if env_keys or info_keys or prob_keys:
            plants = [_to_uniform(p, env_keys, info_keys, prob_keys) for p in plants]
            print("Transformed raw data to uniform schema (environment_care, info, problems)")
        else:
            print("WARNING: Schema not found; skipping uniform transform. Mapping may produce defaults.")

    from resources.ETL.mapping import apply_mappings

    plants = apply_mappings(plants)
    print("Applied mappings (light, water, soil, layer, etc.)")

    # Write mapped plants to JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"plants": plants}, f, indent=2)
    print(f"Wrote {len(plants)} plants to {OUTPUT_PATH}")

    if "--skip-embed" not in sys.argv:
        embed_plants(plants)
    else:
        texts = [plant_to_profile_text(p) for p in plants]
        print("\n--- Example of text being embedded (first plant) ---")
        print(texts[0][:1500] + ("..." if len(texts[0]) > 1500 else ""))
        print("---\n")

    if "--no-upload" in sys.argv:
        print("Skipping upload (--no-upload)")
        return 0

    try:
        from pymongo import MongoClient
    except ImportError:
        print("pymongo not installed. Run: pip install pymongo")
        return 1

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    collection_name = os.getenv("NEW_PLANT_COLLECTION", "NewPlantCollection")

    if not mongo_uri:
        print("MONGO_URI not set in .env. Skipping upload.")
        return 0

    client = MongoClient(mongo_uri)
    coll = client[mongo_db][collection_name]
    coll.delete_many({})
    result = coll.insert_many(plants)
    print(f"Uploaded {len(result.inserted_ids)} plants to {mongo_db}.{collection_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
