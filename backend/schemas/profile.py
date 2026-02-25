"""Profile request/response schemas."""
from pydantic import BaseModel


class ProfileUpdate(BaseModel):
    """Profile fields to update. Username comes from JWT (Authorization header)."""
    # Profile
    name: str | None = None
    avatar_url: str | None = None
    # Location
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str | None = "US"
    # Climate
    climate: str | None = None
    # Environment
    light_level: str | None = None
    humidity_level: str | None = None
    temp_min_f: float | None = None
    temp_max_f: float | None = None
    # Safety
    has_kids: bool | None = None
    # Constraints
    preferred_size: str | None = None
    hard_no: list[str] | None = None
    # Preferences (care_level aligns with plant care_level: easy/medium/hard)
    care_level: str | None = None
    watering_freq: str | None = None
    care_freq: str | None = None
