"""Garden request/response schemas."""
from typing import Any

from pydantic import BaseModel


class AddToGardenRequest(BaseModel):
    plant_id: int
    custom_name: str | None = None


class DeathReport(BaseModel):
    plant_id: int
    what_happened: list[str] = []  # multi-select, optional
    watering_frequency: str | None = None
    plant_location: str | None = None
    humidity_level: str | None = None  # HUMIDITY_VOCAB: low, medium, high
    room_temperature: str | None = None  # cold, comfortable, hot
    death_reason: str | None = None
    plant_profile: dict[str, Any] | None = None
    user_profile: dict[str, Any] | None = None
