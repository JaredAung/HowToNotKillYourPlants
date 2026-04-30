"""
ETL: Data gathering from Permapeople API and mapping to schema.

Uses resources.data_creating.permapeople_plant for API access.

Run from project root:
  python -m resources.data_creating.data_creating_pipeline gather-permapeople
  python -m resources.data_creating.data_creating_pipeline gather-permapeople --all
  python -m resources.data_creating.data_creating_pipeline gather-permapeople --batches 10
  python -m resources.data_creating.data_creating_pipeline map-permapeople       # map + upload to NewPlantCollection
  python -m resources.data_creating.data_creating_pipeline map-permapeople --no-upload
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCES = Path(__file__).resolve().parent.parent

# Ensure project root is on path for resources.* imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# Paths
PERMAPEOPLE_OUTPUT = RESOURCES / "data_creating" / "permapeople_plants_sample.json"
PERMAPEOPLE_MAPPED = RESOURCES / "data_creating" / "permapeople_plants_mapped.json"
PERMAPEOPLE_DATA_KEYS_SCHEMA = RESOURCES / "data_creating" / "permapeople_data_keys_schema.json"
PERMAPEOPLE_SAMPLE_SCHEMA = RESOURCES / "schema" / "permapeople_plants_sample.schema.json"


def gather_permapeople(
    *,
    all_plants: bool = False,
    batches: int | None = None,
    output_path: Path | None = None,
) -> list[dict]:
    """
    Gather plant data from the Permapeople API.

    Args:
        all_plants: If True, fetch all plants (paginate until empty).
        batches: If set, fetch this many batches of 100 plants (append to existing).
        output_path: Where to save. Default: resources/permapeople_plants_sample.json.

    Returns:
        List of plant objects.
    """
    from resources.data_creating.permapeople_plant import list_plants, fetch_all_plants

    out = output_path or PERMAPEOPLE_OUTPUT
    existing: list[dict] = []
    seen_ids: set[int] = set()

    if out.exists():
        with open(out) as f:
            data = json.load(f)
            existing = data.get("plants", [])
            seen_ids = {p["id"] for p in existing if p.get("id") is not None}
        print(f"Loaded {len(existing)} existing plants from {out}")

    if all_plants:
        print("Fetching all plants from Permapeople API...")
        plants = fetch_all_plants()
        # Merge: keep existing, add new by id
        for p in plants:
            pid = p.get("id")
            if pid is not None and pid not in seen_ids:
                existing.append(p)
                seen_ids.add(pid)
        print(f"Total plants: {len(existing)}")
    elif batches is not None and batches > 0:
        last_id = existing[-1].get("id") if existing else None
        for i in range(batches):
            print(f"Fetching batch {i + 1}/{batches} (last_id={last_id})...")
            batch = list_plants(last_id=last_id)
            if not batch:
                print("  No more plants.")
                break
            new_count = 0
            for p in batch:
                pid = p.get("id")
                if pid is not None and pid not in seen_ids:
                    existing.append(p)
                    seen_ids.add(pid)
                    new_count += 1
            print(f"  Got {len(batch)} plants, {new_count} new")
            last_id = batch[-1].get("id") if batch else None
            if last_id is None:
                break
    else:
        # Default: fetch first page only
        print("Fetching first batch (100 plants)...")
        batch = list_plants()
        for p in batch:
            pid = p.get("id")
            if pid is not None and pid not in seen_ids:
                existing.append(p)
                seen_ids.add(pid)
        print(f"Got {len(batch)} plants, total {len(existing)}")

    with open(out, "w") as f:
        json.dump({"plants": existing}, f, indent=2)
    print(f"Saved to {out}")

    return existing


def map_plant(plant: dict, key_to_category: dict, excluded_keys: set) -> dict:
    """Transform a plant record: categorize data, remove parent_id/adopter_id/version."""
    data_dict = {}
    for item in plant.get("data", []):
        key = item.get("key")
        value = item.get("value")
        if key and key not in excluded_keys:
            data_dict[key] = value

    environment_care = {}
    info = {}
    problems = {}

    for key, value in data_dict.items():
        cat = key_to_category.get(key)
        if cat == "environment_care":
            environment_care[key] = value
        elif cat == "info":
            info[key] = value
        elif cat == "problems":
            problems[key] = value

    return {
        "id": plant.get("id"),
        "name": plant.get("name"),
        "slug": plant.get("slug"),
        "scientific_name": plant.get("scientific_name"),
        "description": plant.get("description"),
        "environment_care": environment_care,
        "info": info,
        "problems": problems,
        "created_at": plant.get("created_at"),
        "updated_at": plant.get("updated_at"),
        "type": plant.get("type"),
        "link": plant.get("link"),
        "images": plant.get("images"),
    }


def _load_mapping_schema(schema_path: Path) -> tuple[dict, set]:
    """Load key_to_category and excluded_keys from schema. Supports both schema formats."""
    with open(schema_path) as f:
        schema_data = json.load(f)

    key_to_category = schema_data.get("key_to_category")
    excluded_keys = set(schema_data.get("excluded_keys", ["Danish name", "Dutch name", "German name"]))

    if key_to_category is None:
        # Build from permapeople_plants_sample.schema (environment, info, problems)
        key_to_category = {}
        for key in schema_data.get("environment", []):
            key_to_category[key] = "environment_care"
        for key in schema_data.get("info", []):
            key_to_category[key] = "info"
        for key in schema_data.get("problems", []):
            key_to_category[key] = "problems"

    return key_to_category, excluded_keys


def map_permapeople(
    input_path: Path | None = None,
    schema_path: Path | None = None,
    output_path: Path | None = None,
    upload: bool = True,
) -> list[dict]:
    """
    Map permapeople_plants_sample.json to schema (environment_care, info, problems).
    Removes parent_id, adopter_id, version. Filters excluded keys.
    """
    inp = input_path or PERMAPEOPLE_OUTPUT
    out = output_path or PERMAPEOPLE_MAPPED

    schema = schema_path or (
        PERMAPEOPLE_DATA_KEYS_SCHEMA
        if PERMAPEOPLE_DATA_KEYS_SCHEMA.exists()
        else PERMAPEOPLE_SAMPLE_SCHEMA
    )

    with open(inp) as f:
        data = json.load(f)

    key_to_category, excluded_keys = _load_mapping_schema(schema)

    plants = data.get("plants", [])
    mapped = [map_plant(p, key_to_category, excluded_keys) for p in plants]

    with open(out, "w") as f:
        json.dump({"plants": mapped}, f, indent=2)
    print(f"Mapped {len(mapped)} plants to {out}")

    if upload:
        upload_mapped_to_mongo(mapped_path=out)
    return mapped


def upload_mapped_to_mongo(
    mapped_path: Path | None = None,
    dry_run: bool = False,
) -> int:
    """
    Upload mapped plants from permapeople_plants_mapped.json to MongoDB NewPlantCollection.
    Replaces all documents in the collection.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        print("pymongo not installed. Run: pip install pymongo")
        return 0

    path = mapped_path or PERMAPEOPLE_MAPPED
    if not path.exists():
        print(f"Mapped file not found: {path}. Run map-permapeople first.")
        return 0

    with open(path) as f:
        data = json.load(f)
    plants = data.get("plants", [])
    if not plants:
        print("No plants to upload.")
        return 0

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    collection_name = os.getenv("NEW_PLANT_COLLECTION", "NewPlantCollection")

    if not mongo_uri:
        print("MONGO_URI not set in .env. Skipping upload.")
        return 0

    if dry_run:
        print(f"DRY RUN: Would upload {len(plants)} plants to {mongo_db}.{collection_name}")
        return len(plants)

    client = MongoClient(mongo_uri)
    coll = client[mongo_db][collection_name]
    coll.delete_many({})
    result = coll.insert_many(plants)
    print(f"Uploaded {len(result.inserted_ids)} plants to {mongo_db}.{collection_name}")
    return len(result.inserted_ids)


def main():
    sys.path.insert(0, str(ROOT))

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    if cmd == "map-permapeople":
        do_upload = "--no-upload" not in sys.argv
        map_permapeople(upload=do_upload)
        return
    if cmd != "gather-permapeople":
        print(__doc__)
        return

    all_plants = "--all" in sys.argv
    batches = None
    for i, arg in enumerate(sys.argv):
        if arg == "--batches" and i + 1 < len(sys.argv):
            try:
                batches = int(sys.argv[i + 1])
            except ValueError:
                batches = None
            break

    gather_permapeople(all_plants=all_plants, batches=batches)


if __name__ == "__main__":
    main()
