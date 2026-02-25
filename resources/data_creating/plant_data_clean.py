"""
Load house_plants_enriched_schema.json and transform into Plant Profile schema.
"""
import json
import re
from pathlib import Path

from dotenv import load_dotenv
import voyageai

# Load .env from project root (parent of resources/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SCHEMA_PATH = Path(__file__).parent / "house_plants_enriched_schema.json"
OUTPUT_PATH = Path(__file__).parent / "plant_profiles.json"
EMBED_MODEL = "voyage-4-lite"  # cost-optimized; try voyage-4 or voyage-4-large if unavailable
BATCH_SIZE = 128


def parse_light_duration(text: str | None) -> str | None:
    """Extract duration from light description if present."""
    if not text:
        return None
    # e.g. "6 or more hours of direct sunlight per day"
    m = re.search(r"(\d+)\s*(?:or\s+more)?\s*hours?", text, re.I)
    if m:
        num = m.group(1)
        return f"{num}+ hours" if "or more" in text.lower() else f"{num} hours"
    return None


def to_fahrenheit(celsius: float | None, fahrenheit: float | None) -> float | None:
    """Return temperature in Fahrenheit. Prefer celsius conversion when available (source fahrenheit can be wrong)."""
    if celsius is not None:
        return round(celsius * 9 / 5 + 32, 1)
    if fahrenheit is not None:
        return float(fahrenheit)
    return None


def bucket_water_req(text: str | None) -> str:
    """Classify watering requirement into high/medium/low bucket (aligned with user watering_freq)."""
    if not text:
        return "medium"
    t = text.lower()
    if "must not be dry" in t or "keep moist" in t:
        return "high"
    if "only when dry" in t or "must be dry" in t:
        return "low"
    if "half dry" in t:
        return "medium"
    return "medium"


def normalize_climate(raw: str | None) -> str | None:
    """Normalize climate for user alignment. 'Tropical to subtropical' -> 'Tropical'."""
    if not raw:
        return None
    if raw.strip() == "Tropical to subtropical":
        return "Tropical"
    return raw


def bucket_humidity_req(text: str | None) -> str | None:
    """Classify humidity requirement into low/medium/high (snake_case convention)."""
    if not text:
        return None
    t = text.lower()
    if re.search(r"moderate\s+to\s+high", t):
        return "medium"
    if re.search(r"\bhigh\b", t):
        return "high"
    if re.search(r"\bmoderate\b", t):
        return "medium"
    if re.search(r"\blow\b", t):
        return "low"
    return None


def normalize_light(raw: str | None) -> str | None:
    """Map raw light description to snake_case: direct, bright_light, bright_indirect, indirect, diffused."""
    if not raw or raw.strip() == "/":
        return None
    t = raw.lower()
    if "direct" in t and ("6" in t or "hours" in t or "sunlight" in t):
        return "direct"
    if "direct sunlight" in t or "direct sun" in t:
        return "direct"
    if "bright" in t and "indirect" in t:
        return "bright_indirect"
    if "bright light" in t or t == "bright light":
        return "bright_light"
    if "indirect" in t and "bright" not in t:
        return "indirect"
    if "diffused" in t or "diffuse" in t:
        return "diffused"
    if "bright" in t:
        return "bright_light"  # fallback
    return None


def extract_soil(soil_text: str | None) -> dict:
    """Extract structured soil fields from description using regex."""
    if not soil_text:
        return {
            "soil_type": None,
            "drainage_level": None,
            "plant_specialty": None,
            "texture": None,
            "moisture_retention": None,
            "organic_content": None,
            "special_notes": [],
        }
    s = soil_text.lower()

    # drainage_level (low/medium/high convention)
    if "fast-draining" in s or "sharp drainage" in s:
        drainage_level = "fast"
    elif "well-draining" in s and ("moisture-retentive" in s or "moist but" in s):
        drainage_level = "medium"
    elif "well-draining" in s or "excellent drainage" in s:
        drainage_level = "well"
    elif "moisture-retentive" in s or "moist but" in s:
        drainage_level = "medium"
    else:
        drainage_level = None

    # plant_specialty (check specific before generic)
    if "no soil required" in s or "mounted or placed" in s:
        plant_specialty = "none"
    elif re.search(r"\baroid\b", s):
        plant_specialty = "aroid"
    elif "bromeliad" in s:
        plant_specialty = "bromeliad"
    elif "cactus" in s or "succulent" in s:
        plant_specialty = "cactus"  # umbrella for both
    elif "fern" in s:
        plant_specialty = "fern"
    elif "orchid" in s:
        plant_specialty = "orchid"
    elif "palm" in s:
        plant_specialty = "palm"
    elif "houseplant" in s or "potting" in s or "standard" in s:
        plant_specialty = "houseplant"
    else:
        plant_specialty = None

    # texture
    if "chunky" in s:
        texture = "chunky"
    elif "loose" in s:
        texture = "loose"
    elif "airy" in s:
        texture = "airy"
    elif "sandy" in s or "slightly sandy" in s:
        texture = "sandy"
    elif "peat-based" in s or "peat and" in s:
        texture = "peat-based"
    elif "bark" in s and "orchid" in s:
        texture = "bark"
    elif "bark" in s:
        texture = "bark"
    else:
        texture = "standard" if drainage_level else None

    # moisture_retention (low/medium/high convention)
    if "moisture-retentive" in s:
        moisture_retention = "high"
    elif "moist but" in s:
        moisture_retention = "medium"
    else:
        moisture_retention = "standard" if drainage_level else None

    # organic_content
    if re.search(r"\brich\b", s) or "organic matter" in s or "organic soil" in s:
        organic_content = "rich"
    else:
        organic_content = None

    # special_notes
    special_notes = []
    if "mounted" in s or "mount" in s:
        special_notes.append("mounted")
    if "bark" in s:
        special_notes.append("bark")
    if "perlite" in s:
        special_notes.append("perlite")
    if "no soil" in s:
        special_notes.append("no_soil")
    if "organic" in s and organic_content is None:
        special_notes.append("organic")

    return {
        "soil_type": soil_text,
        "drainage_level": drainage_level,
        "plant_specialty": plant_specialty,
        "texture": texture,
        "moisture_retention": moisture_retention,
        "organic_content": organic_content,
        "special_notes": special_notes,
    }


def concat_desc(profile: dict) -> str:
    """Concatenate physical_desc and symbolism for embedding."""
    desc = profile.get("Info", {}).get("desc", {}) or {}
    parts = [
        desc.get("physical_desc") or "",
        desc.get("symbolism") or "",
    ]
    text = " ".join(p.strip() for p in parts if p).strip()
    return text or "No description"


def embed_descriptions(profiles: list[dict]) -> list[list[float]]:
    """Embed concatenated descriptions using Voyage AI. Returns list of embedding vectors."""
    texts = [concat_desc(p) for p in profiles]
    vo = voyageai.Client()
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = vo.embed(batch, model=EMBED_MODEL, input_type="document")
        all_embeddings.extend(result.embeddings)
    return all_embeddings


def transform_plant(raw: dict) -> dict:
    """Transform a raw plant record into the Plant Profile schema."""
    common = raw.get("common") or []
    common_name = ", ".join(common) if isinstance(common, list) else str(common)

    desc = raw.get("description") or {}
    soil_list = raw.get("soil_type") or []
    soil_text = soil_list[0] if soil_list else None

    ideal_light_raw = raw.get("ideallight") or ""
    tolerated_light_raw = raw.get("toleratedlight") or ""
    if tolerated_light_raw == "/":
        tolerated_light_raw = ""

    tempmax = raw.get("tempmax") or {}
    tempmin = raw.get("tempmin") or {}

    return {
        "plant_id": raw.get("id"),
        "img_url": raw.get("image_url"),
        "Info": {
            "latin": raw.get("latin"),
            "common_name": common_name or None,
            "category": raw.get("category"),
            "origin": raw.get("origin"),
            "size": raw.get("size_bucket") or (f"{raw.get('size_max_cm')} cm" if raw.get("size_max_cm") else None),
            "desc": {
                "physical_desc": desc.get("physical"),
                "symbolism": desc.get("symbolism"),
            },
            "growth_rate": raw.get("growth_rate"),
        },
        "Care": {
            "water_req": raw.get("watering"),
            "water_req_bucket": bucket_water_req(raw.get("watering")),
            "light_req": {
                "ideal_light": {
                    "sunlight_type": ideal_light_raw or None,
                    "sunlight_bucket": normalize_light(ideal_light_raw),
                    "duration": parse_light_duration(ideal_light_raw),
                },
                "tolerated_light": {
                    "sunlight_type": tolerated_light_raw or None,
                    "sunlight_bucket": normalize_light(tolerated_light_raw),
                    "duration": parse_light_duration(tolerated_light_raw),
                },
            },
            "humidity_req": (raw.get("care_guidelines") or {}).get("humidity_detail"),
            "humidity_req_bucket": bucket_humidity_req((raw.get("care_guidelines") or {}).get("humidity_detail")),
            "temp_req": {
                "min_temp": to_fahrenheit(tempmin.get("celsius"), tempmin.get("fahrenheit")),
                "max_temp": to_fahrenheit(tempmax.get("celsius"), tempmax.get("fahrenheit")),
            },
            "climate": normalize_climate(raw.get("climate")),
            "disease": raw.get("diseases") or [],
            "bugs": raw.get("insects") or [],
            "soil": extract_soil(soil_text),
            "care_level": raw.get("care_level"),
        },
    }


def main(skip_embeddings: bool = False):
    with open(SCHEMA_PATH) as f:
        plants = json.load(f)

    profiles = [transform_plant(p) for p in plants]

    if skip_embeddings:
        print("Skipping embeddings (--skip-embeddings)")
        if OUTPUT_PATH.exists():
            with open(OUTPUT_PATH) as f:
                old = json.load(f)
            for p, old_p in zip(profiles, old):
                p["desc_embeddings"] = old_p.get("desc_embeddings", [])
        else:
            for p in profiles:
                p["desc_embeddings"] = []
    else:
        print(f"Embedding {len(profiles)} plant descriptions...")
        embeddings = embed_descriptions(profiles)
        for profile, emb in zip(profiles, embeddings):
            profile["desc_embeddings"] = emb

    with open(OUTPUT_PATH, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"Transformed {len(profiles)} plants -> {OUTPUT_PATH}")
    return profiles


if __name__ == "__main__":
    import sys
    skip = "--skip-embeddings" in sys.argv
    main(skip_embeddings=skip)
