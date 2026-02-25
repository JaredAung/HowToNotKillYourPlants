"""Auth request/response schemas."""
from pydantic import BaseModel


class SignUpRequest(BaseModel):
    username: str
    email: str | None = None
    password: str


class LogInRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    message: str
    username: str | None = None
    token: str | None = None  # JWT for subsequent requests
