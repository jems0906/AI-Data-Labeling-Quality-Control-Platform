"""Authentication and local user-store utilities for the labeling QC dashboard."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
USER_DB_PATH = DATA_DIR / "users.json"
AUDIT_LOG_PATH = DATA_DIR / "audit_log.json"
ROLES = ["Administrator", "Operations Manager", "Customer Success"]
DEFAULT_USERS = {
    "admin": {
        "password": "admin123",
        "role": "Administrator",
        "display_name": "Platform Admin",
        "active": True,
    },
    "ops": {
        "password": "ops123",
        "role": "Operations Manager",
        "display_name": "Ops Manager",
        "active": True,
    },
    "client": {
        "password": "client123",
        "role": "Customer Success",
        "display_name": "Client Success",
        "active": True,
    },
}


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_username(username: str) -> str:
    return username.strip().lower()


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    ).hex()
    return digest, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    digest, _ = hash_password(password, salt)
    return hmac.compare_digest(digest, stored_hash)


def build_user_record(
    username: str,
    display_name: str,
    role: str,
    password: str,
    active: bool = True,
) -> dict[str, Any]:
    password_hash, salt = hash_password(password)
    timestamp = now_iso()
    return {
        "username": normalize_username(username),
        "display_name": display_name.strip(),
        "role": role,
        "active": bool(active),
        "password_hash": password_hash,
        "salt": salt,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_users(users: dict[str, dict[str, Any]]) -> None:
    ordered = [users[key] for key in sorted(users)]
    write_json(USER_DB_PATH, {"users": ordered})


def ensure_storage() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if not USER_DB_PATH.exists():
        seeded_users = [
            build_user_record(
                username=username,
                display_name=config["display_name"],
                role=config["role"],
                password=config["password"],
                active=config["active"],
            )
            for username, config in DEFAULT_USERS.items()
        ]
        write_json(USER_DB_PATH, {"users": seeded_users})

    if not AUDIT_LOG_PATH.exists():
        write_json(
            AUDIT_LOG_PATH,
            [
                {
                    "timestamp": now_iso(),
                    "actor": "system",
                    "action": "Initialized local authentication store",
                    "status": "Success",
                }
            ],
        )


def load_users() -> dict[str, dict[str, Any]]:
    ensure_storage()
    payload = json.loads(USER_DB_PATH.read_text(encoding="utf-8"))
    users = {item["username"]: item for item in payload.get("users", [])}

    if not any(user.get("role") == "Administrator" for user in users.values()):
        admin = DEFAULT_USERS["admin"]
        users["admin"] = build_user_record(
            username="admin",
            display_name=admin["display_name"],
            role=admin["role"],
            password=admin["password"],
            active=True,
        )
        save_users(users)

    return users


def load_audit_log() -> pd.DataFrame:
    ensure_storage()
    events = json.loads(AUDIT_LOG_PATH.read_text(encoding="utf-8"))
    frame = pd.DataFrame(events)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "actor", "action", "status"])
    return frame.sort_values("timestamp", ascending=False)


def append_audit_event(actor: str, action: str, status: str = "Success") -> None:
    ensure_storage()
    events = json.loads(AUDIT_LOG_PATH.read_text(encoding="utf-8"))
    events.append(
        {
            "timestamp": now_iso(),
            "actor": actor,
            "action": action,
            "status": status,
        }
    )
    write_json(AUDIT_LOG_PATH, events[-100:])


def set_authenticated_user(user: dict[str, Any]) -> None:
    st.session_state.authenticated = True
    st.session_state.username = user["username"]
    st.session_state.role = user["role"]
    st.session_state.display_name = user.get("display_name") or user["username"]


def authenticate_user(username: str, password: str) -> tuple[bool, str | None, dict[str, Any] | None]:
    normalized = normalize_username(username)
    users = load_users()
    user = users.get(normalized)

    if user is None:
        append_audit_event(normalized or "unknown", "Login attempt", "Failed - user not found")
        return False, "User not found.", None

    if not user.get("active", True):
        append_audit_event(normalized, "Login attempt", "Failed - account disabled")
        return False, "This account is disabled.", None

    if verify_password(password, user["password_hash"], user["salt"]):
        append_audit_event(normalized, "Signed in", "Success")
        return True, None, user

    append_audit_event(normalized, "Login attempt", "Failed - incorrect password")
    return False, "Incorrect password.", None
