"""
Embed synthetic user profiles with Voyage AI and add profile_embedding to each user.
Run: python -m backend.eval.embed
"""
import json
from pathlib import Path

from dotenv import load_dotenv
import voyageai

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

USERS_PATH = ROOT / "resources" / "synthetic_users.json"
EMBED_MODEL = "voyage-4-lite"
BATCH_SIZE = 128


def profile_to_text(user: dict) -> str:
    """Concatenate user profile into a single text for embedding."""
    parts = []
    if user.get("climate"):
        parts.append(f"climate: {user['climate']}")
    env = user.get("environment") or {}
    if env:
        env_parts = []
        if env.get("light_level"):
            env_parts.append(f"light_level: {env['light_level']}")
        if env.get("humidity_level"):
            env_parts.append(f"humidity_level: {env['humidity_level']}")
        temp = env.get("temperature_pref") or {}
        if temp.get("min_f") is not None or temp.get("max_f") is not None:
            env_parts.append(f"temp: {temp.get('min_f')}-{temp.get('max_f')}°F")
        if env_parts:
            parts.append("environment: " + ", ".join(env_parts))
    constraints = user.get("constraints") or {}
    if constraints.get("preferred_size"):
        parts.append(f"constraints: preferred_size: {constraints['preferred_size']}")
    pref = user.get("preferences") or {}
    if pref:
        pref_parts = []
        if pref.get("care_level"):
            pref_parts.append(f"care_level: {pref['care_level']}")
        care_pref = pref.get("care_preferences") or {}
        if care_pref.get("watering_freq"):
            pref_parts.append(f"watering_freq: {care_pref['watering_freq']}")
        if care_pref.get("care_freq"):
            pref_parts.append(f"care_freq: {care_pref['care_freq']}")
        if pref_parts:
            parts.append("preferences: " + ", ".join(pref_parts))
    return " | ".join(parts) if parts else "no profile"


def main():
    print(f"Loading {USERS_PATH}...")
    with open(USERS_PATH) as f:
        users = json.load(f)
    print(f"  {len(users)} users")

    texts = [profile_to_text(u) for u in users]
    print("Embedding with Voyage AI...")
    vo = voyageai.Client()
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = vo.embed(batch, model=EMBED_MODEL, input_type="document")
        all_embeddings.extend(result.embeddings)
        print(f"  embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    for user, emb in zip(users, all_embeddings):
        user["profile_embedding"] = emb

    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=2)
    print(f"Saved {USERS_PATH} with profile_embedding for each user.")


if __name__ == "__main__":
    main()
