"""Pydantic schemas for request/response validation."""
from schemas.auth import SignUpRequest, LogInRequest, AuthResponse
from schemas.garden import AddToGardenRequest, DeathReport
from schemas.profile import ProfileUpdate

__all__ = [
    "SignUpRequest",
    "LogInRequest",
    "AuthResponse",
    "AddToGardenRequest",
    "DeathReport",
    "ProfileUpdate",
]
