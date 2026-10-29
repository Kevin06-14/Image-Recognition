"""JWT authentication and password storage for the web service."""

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv

from src.database import _connect, init_db


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
JWT_SECRET = os.getenv("AUTH_SECRET", "development-only-change-this-secret").encode(
    "utf-8"
)


def init_auth_db() -> None:
    init_db()
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def _password_hash(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120000)
    return base64.urlsafe_b64encode(salt + digest).decode("ascii")


def _verify_password(password: str, stored: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(stored.encode("ascii"))
        salt, expected = raw[:16], raw[16:]
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120000)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def create_user(username: str, password: str) -> None:
    username = username.strip()
    if len(username) < 3 or len(password) < 6:
        raise ValueError("用户名至少 3 个字符，密码至少 6 个字符。")
    with _connect() as connection:
        try:
            connection.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (
                    username,
                    _password_hash(password, secrets.token_bytes(16)),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("用户名已经存在。") from exc


def authenticate_user(username: str, password: str) -> bool:
    with _connect() as connection:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    return bool(row and _verify_password(password, row["password_hash"]))


def _encode(value: Dict[str, Any]) -> str:
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def create_token(username: str) -> str:
    header = _encode({"alg": "HS256", "typ": "JWT"})
    payload = _encode({"sub": username, "exp": int(time.time()) + 8 * 60 * 60})
    message = f"{header}.{payload}".encode("ascii")
    signature = hmac.new(JWT_SECRET, message, hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{header}.{payload}.{encoded_signature}"


def decode_token(token: str) -> str:
    try:
        header, payload, signature = token.split(".")
        message = f"{header}.{payload}".encode("ascii")
        expected = hmac.new(JWT_SECRET, message, hashlib.sha256).digest()
        actual = base64.urlsafe_b64decode(signature + "===")
        if not hmac.compare_digest(actual, expected):
            raise ValueError("令牌签名无效。")
        data = json.loads(base64.urlsafe_b64decode(payload + "==="))
        if int(data["exp"]) < int(time.time()):
            raise ValueError("登录令牌已过期，请重新登录。")
        return str(data["sub"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("登录令牌无效，请重新登录。") from exc
