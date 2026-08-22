# app/utils/jwt.py
import jwt
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "8d7f6e4a2b1c9f0d8e7c6b5a4d3f2g1h")
ALGORITHM = "HS256"


def create_access_token(data: dict, expires_minutes: int = 60) -> str:
    """
    Create a JWT token with payload `data` and expiration in `expires_minutes`.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict | None:
    """
    Decode and verify a JWT token. Returns payload if valid, else None.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def verify_token(token: str) -> dict | None:
    """
    Alias for decode_access_token for backward compatibility.
    """
    return decode_access_token(token)