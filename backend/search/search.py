"""
Search API: extract profile fields from free text using LLM.
Merges with user's existing profile. Does not persist to MongoDB.
"""
import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.jwt import get_current_username
from database import get_user_collection
from llm import chat_simple
from recommend.feature_loader import normalize_profile
from recommend.recommend import recommend_for_profile

router = APIRouter(prefix="/search", tags=["search"])

EXTRACT_SCHEMA = """
Return a JSON object with ONLY the fields you can infer from the user's text.
Use these exact keys. Omit any field you cannot infer.

Structured fields (use exact values):
- light_level: one of "direct", "bright_light", "bright_indirect", "indirect", "diffused"
- humidity_level: one of "low", "medium", "high"
- temp_min_f: number (Fahrenheit)
- temp_max_f: number (Fahrenheit)
- climate: one of "Arid Tropical", "Subtropical", "Subtropical arid", "Tropical", "Tropical humid"
- preferred_size: one of "small", "medium", "large"
- care_level: one of "easy", "medium", "hard"
- watering_freq: one of "low", "medium", "high"
- care_freq: one of "low", "medium", "high"

Free-text fields (user describing what kind of plant they want):
- physical_desc: string - appearance, shape, foliage, color, texture (e.g. "tall palm with elegant fronds", "glossy heart-shaped leaves")
- symbolism: string - meaning, vibe, feeling (e.g. "peace and tranquility", "good luck and prosperity")

Example: {"physical_desc": "tall palm with feathery fronds", "symbolism": "peace and balance"}
Example: {"light_level": "bright_indirect", "physical_desc": "trailing vine with heart-shaped leaves"}
Return ONLY valid JSON, no markdown or extra text.
"""


def _get_user_profile(username: str) -> dict:
    """Fetch user profile from MongoDB."""
    collection = get_user_collection()
    user = collection.find_one({"auth.username": username}) or collection.find_one(
        {"auth.email": username.lower()}
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "username": (user.get("auth") or {}).get("username") or username,
        "profile": user.get("profile") or {},
        "location": user.get("location") or {},
        "environment": user.get("environment") or {},
        "climate": user.get("climate"),
        "safety": user.get("safety") or {},
        "constraints": user.get("constraints") or {},
        "preferences": user.get("preferences") or {},
    }


def _parse_extracted(text: str) -> dict:
    """Parse LLM JSON response. Returns empty dict on failure."""
    text = text.strip()
    # Remove markdown code blocks if present
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    else:
        # Try to find JSON object
        start = text.find("{")
        if start >= 0:
            end = text.rfind("}") + 1
            if end > start:
                text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _merge_profile(existing: dict, extracted: dict) -> dict:
    """Merge extracted fields into existing profile. Extracted overrides existing."""
    result = dict(existing)
    env = dict(result.get("environment") or {})
    pref = dict(result.get("preferences") or {})
    care_pref = dict(pref.get("care_preferences") or {})
    constraints = dict(result.get("constraints") or {})
    temp = dict(env.get("temperature_pref") or {})

    if extracted.get("light_level") is not None:
        env["light_level"] = extracted["light_level"]
    if extracted.get("humidity_level") is not None:
        env["humidity_level"] = extracted["humidity_level"]
    if extracted.get("temp_min_f") is not None:
        temp["min_f"] = float(extracted["temp_min_f"])
    if extracted.get("temp_max_f") is not None:
        temp["max_f"] = float(extracted["temp_max_f"])
    if temp:
        env["temperature_pref"] = temp

    if extracted.get("climate") is not None:
        result["climate"] = extracted["climate"]
    if extracted.get("preferred_size") is not None:
        constraints["preferred_size"] = extracted["preferred_size"]
    if extracted.get("care_level") is not None:
        pref["care_level"] = extracted["care_level"]
    if extracted.get("watering_freq") is not None:
        care_pref["watering_freq"] = extracted["watering_freq"]
    if extracted.get("care_freq") is not None:
        care_pref["care_freq"] = extracted["care_freq"]

    if extracted.get("physical_desc") is not None and str(extracted["physical_desc"]).strip():
        result["physical_desc"] = str(extracted["physical_desc"]).strip()
    if extracted.get("symbolism") is not None and str(extracted["symbolism"]).strip():
        result["symbolism"] = str(extracted["symbolism"]).strip()

    pref["care_preferences"] = care_pref
    result["environment"] = env
    result["preferences"] = pref
    result["constraints"] = constraints
    return result


class SearchExtractBody(BaseModel):
    text: str = ""


@router.post("/extract")
def extract_profile(
    body: SearchExtractBody,
    username: str = Depends(get_current_username),
):
    """
    Extract profile fields from free text using LLM.
    Merges with user's existing profile. Does NOT save to MongoDB.
    Returns the merged profile for display only.
    """
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    existing = _get_user_profile(username)

    system = (
        "You extract plant preferences from user descriptions. "
        "The text may describe: (1) environment/care needs (light, humidity, etc.), "
        "or (2) what kind of plant they want - physical appearance (shape, leaves, color) and symbolism (meaning, vibe). "
        "Extract both structured fields and free-text physical_desc/symbolism when present. "
        "Infer only what is clearly stated or strongly implied. "
        + EXTRACT_SCHEMA
    )
    user_msg = (
        f"User's current profile (for context): {json.dumps(existing, default=str)}\n\n"
        f"User's new text:\n{text}\n\n"
        "Extract any plant preference fields from the new text. Return JSON."
    )

    try:
        out = chat_simple(user_message=user_msg, system=system)
        extracted = _parse_extracted(out or "")
    except Exception as e:
        logging.warning("Search extract LLM failed: %s", e)
        extracted = {}

    merged = _merge_profile(existing, extracted)
    normalized = normalize_profile(merged)
    username = normalized.get("username", existing.get("username", ""))
    # Use merged+normalized profile (NOT from MongoDB) for recommendations
    rec = recommend_for_profile(normalized, username, k=20)
    return {
        "profile": normalized,
        "plants": rec["plants"],
        "extracted_keys": list(extracted.keys()),
    }
