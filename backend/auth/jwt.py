"""
JWT helpers: create tokens and get current username from Authorization header.
"""
import hashlib
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7

security = HTTPBearer(auto_error=False)


def _token_hash(token: str) -> str:
    """SHA256 hash of token for blacklist storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def is_token_blacklisted(token: str) -> bool:
    """Check if token has been revoked (logged out)."""
    from database import get_token_blacklist_collection
    coll = get_token_blacklist_collection()
    h = _token_hash(token)
    return coll.find_one({"token_hash": h}) is not None


def blacklist_token(token: str, exp_ts: int) -> None:
    """Add token to blacklist. exp_ts is Unix timestamp from JWT."""
    from database import get_token_blacklist_collection
    coll = get_token_blacklist_collection()
    exp_dt = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
    coll.insert_one({"token_hash": _token_hash(token), "exp": exp_dt})


def create_token(username: str) -> str:
    """Create a JWT with username. Expires in JWT_EXPIRE_DAYS."""
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """Extract Bearer token from Authorization header. Raises 401 if missing."""
    if not credentials or credentials.scheme != "Bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    return credentials.credentials


def get_current_username(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """
    Extract and verify JWT from Authorization: Bearer <token>.
    Returns the username. Raises 401 if missing or invalid.
    """
    if not credentials or credentials.scheme != "Bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = credentials.credentials
    if is_token_blacklisted(token):
        raise HTTPException(status_code=401, detail="Token has been revoked")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
