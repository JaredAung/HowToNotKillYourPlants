"""Profile request/response schemas."""
from pydantic import BaseModel


class ProfileUpdate(BaseModel):
    """
    Tower fields + optional display name. Login username comes from JWT; ``name`` is optional greeting / display only.
    """

    name: str | None = None
    climate: str | None = None
    light_level: str | None = None
    soil_preference: str | None = None
    temp_min_f: float | None = None
    temp_max_f: float | None = None
    preferred_size: str | None = None
    care_level: str | None = None
    growth_pref: str | None = None
    watering_freq: str | None = None
    usda_zone_min: int | None = None
    usda_zone_max: int | None = None
