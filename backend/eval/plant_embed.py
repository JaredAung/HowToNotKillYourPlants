"""
Embed plant profiles with Voyage AI and add profile_embedding to each plant.
Run: python -m backend.eval.plant_embed
"""
import json
from pathlib import Path

from dotenv import load_dotenv
import voyageai

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

PLANTS_PATH = ROOT / "resources" / "plant_profiles.json"
EMBED_MODEL = "voyage-4-lite"
BATCH_SIZE = 128


def plant_to_text(plant: dict) -> str:
    """Concatenate full plant profile into a single text for embedding."""
    parts = []
    info = plant.get("Info") or {}
    care = plant.get("Care") or {}

    # Info: latin, common_name, category, origin, size, growth_rate
    if info.get("latin"):
        parts.append(f"latin: {info['latin']}")
    if info.get("common_name"):
        parts.append(f"common_name: {info['common_name']}")
    if info.get("category"):
        parts.append(f"category: {info['category']}")
    if info.get("origin"):
        parts.append(f"origin: {info['origin']}")
    if info.get("size"):
        parts.append(f"size: {info['size']}")
    if info.get("growth_rate"):
        parts.append(f"growth_rate: {info['growth_rate']}")

    # Info.desc: physical_desc, symbolism
    desc = info.get("desc") or {}
    if desc.get("physical_desc"):
        parts.append(f"physical_desc: {desc['physical_desc']}")
    if desc.get("symbolism"):
        parts.append(f"symbolism: {desc['symbolism']}")

    # Care: climate, water, light, humidity, temp, soil, care_level, disease, bugs
    if care.get("climate"):
        parts.append(f"climate: {care['climate']}")

    if care.get("water_req"):
        parts.append(f"water_req: {care['water_req']}")
    if care.get("water_req_bucket"):
        parts.append(f"water_req_bucket: {care['water_req_bucket']}")

    light_req = care.get("light_req") or {}
    ideal = light_req.get("ideal_light") or {}
    tolerated = light_req.get("tolerated_light") or {}
    light_parts = []
    if ideal.get("sunlight_type"):
        light_parts.append(f"ideal: {ideal['sunlight_type']}")
    if tolerated.get("sunlight_type"):
        light_parts.append(f"tolerated: {tolerated['sunlight_type']}")
    if light_parts:
        parts.append("light: " + ", ".join(light_parts))

    if care.get("humidity_req"):
        parts.append(f"humidity_req: {care['humidity_req']}")
    if care.get("humidity_req_bucket"):
        parts.append(f"humidity_req_bucket: {care['humidity_req_bucket']}")

    temp = care.get("temp_req") or {}
    if temp.get("min_temp") is not None or temp.get("max_temp") is not None:
        parts.append(f"temp: {temp.get('min_temp')}-{temp.get('max_temp')}°F")

    if care.get("care_level"):
        parts.append(f"care_level: {care['care_level']}")

    if care.get("disease") and isinstance(care["disease"], list) and care["disease"]:
        parts.append(f"disease: {', '.join(care['disease'])}")
    if care.get("bugs") and isinstance(care["bugs"], list) and care["bugs"]:
        parts.append(f"bugs: {', '.join(care['bugs'])}")

    soil = care.get("soil") or {}
    if soil.get("soil_type"):
        parts.append(f"soil_type: {soil['soil_type']}")
    if soil.get("drainage_level"):
        parts.append(f"drainage: {soil['drainage_level']}")
    if soil.get("texture"):
        parts.append(f"soil_texture: {soil['texture']}")
    if soil.get("moisture_retention"):
        parts.append(f"moisture_retention: {soil['moisture_retention']}")
    if soil.get("special_notes") and isinstance(soil["special_notes"], list) and soil["special_notes"]:
        parts.append(f"soil_notes: {', '.join(soil['special_notes'])}")

    return " | ".join(parts) if parts else "no profile"


def main():
    print(f"Loading {PLANTS_PATH}...")
    with open(PLANTS_PATH) as f:
        plants = json.load(f)
    print(f"  {len(plants)} plants")

    texts = [plant_to_text(p) for p in plants]
    print("Embedding with Voyage AI...")
    vo = voyageai.Client()
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = vo.embed(batch, model=EMBED_MODEL, input_type="document")
        all_embeddings.extend(result.embeddings)
        print(f"  embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    for plant, emb in zip(plants, all_embeddings):
        plant["profile_embedding"] = emb

    with open(PLANTS_PATH, "w") as f:
        json.dump(plants, f, indent=2)
    print(f"Saved {PLANTS_PATH} with profile_embedding for each plant.")


if __name__ == "__main__":
    main()
