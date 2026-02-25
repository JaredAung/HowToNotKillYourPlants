"""
Simple authentication module with sign up and log in.
Uses bcrypt for password hashing and MongoDB UserCollection for storage.
"""
from datetime import datetime

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from pymongo.errors import DuplicateKeyError

from auth.jwt import blacklist_token, create_token, get_current_username, get_current_token
from database import get_user_collection
from schemas import AuthResponse, LogInRequest, SignUpRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse)
def sign_up(request: SignUpRequest):
    """
    Register a new user. Username must be unique.
    Password is hashed with bcrypt before storage.
    """
    if not request.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if not request.password:
        raise HTTPException(status_code=400, detail="Password cannot be empty")

    collection = get_user_collection()
    username = request.username.strip()

    existing = collection.find_one({"auth.username": username})
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed = bcrypt.hashpw(
        request.password.encode("utf-8"),
        bcrypt.gensalt()
    )

    auth_doc = {
        "username": username,
        "password_hash": hashed.decode("utf-8"),
        "created_at": datetime.utcnow(),
        "last_login_at": None,
    }
    if request.email and request.email.strip():
        auth_doc["email"] = request.email.strip().lower()
    try:
        collection.insert_one({
            "auth": auth_doc,
            "profile": {},
            "location": {},
            "climate": None,
            "environment": {},
            "safety": {},
            "constraints": {},
            "preferences": {},
            "gamification": {"care_points": 0, "care_level": 0, "streak_days": 0, "multiplier": 1, "badges": []},
            "social": {"friends": [], "neighborhood_id": None},
            "history": {"owned_plants_count": 0, "deaths_count": 0, "average_health_score": 0, "last_death_reason": None},
        })
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Username already taken")

    token = create_token(username)
    return AuthResponse(message="User created successfully", username=username, token=token)


@router.post("/login", response_model=AuthResponse)
def log_in(request: LogInRequest):
    """Authenticate user with username and password."""
    if not request.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if not request.password:
        raise HTTPException(status_code=400, detail="Password cannot be empty")

    collection = get_user_collection()
    username = request.username.strip()
    user = collection.find_one({"auth.username": username})

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    auth = user.get("auth", {})
    stored_hash = auth.get("password_hash", "")
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode("utf-8")

    if not bcrypt.checkpw(request.password.encode("utf-8"), stored_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    collection.update_one(
        {"auth.username": username},
        {"$set": {"auth.last_login_at": datetime.utcnow()}}
    )

    token = create_token(username)
    return AuthResponse(message="Login successful", username=username, token=token)


@router.get("/me")
def get_me(username: str = Depends(get_current_username)):
    """Return the logged-in user's username. Requires Authorization: Bearer <token>."""
    return {"username": username}


@router.post("/logout")
def log_out(token: str = Depends(get_current_token)):
    """
    Revoke the current token (log out).
    Requires Authorization: Bearer <token>. The token will be blacklisted.
    """
    import jwt
    from auth.jwt import JWT_ALGORITHM, JWT_SECRET
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        exp = payload.get("exp")
        if exp:
            blacklist_token(token, exp)
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        pass  # Already invalid, no need to blacklist
    return {"message": "Logged out successfully"}
