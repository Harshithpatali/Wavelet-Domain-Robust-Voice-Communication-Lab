from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

APP_TITLE = "Wavelet Voice Lab Private Call Signaling"
DEFAULT_TTL_MINUTES = 30
MAX_TTL_MINUTES = 120
MAX_CREATE_PER_HOUR = 5
MAX_AUTH_ATTEMPTS_PER_MINUTE = 5
PASSWORD_HASH_ROUNDS = 180_000
MAX_SIGNAL_BYTES = 64 * 1024

app = FastAPI(title=APP_TITLE, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@dataclass
class Room:
    room_id: str
    salt: bytes
    code_hash: bytes
    expires_at: datetime
    participants: dict[str, WebSocket] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


ROOMS: dict[str, Room] = {}
CREATE_ATTEMPTS: dict[str, deque[float]] = defaultdict(deque)
AUTH_ATTEMPTS: dict[tuple[str, str], deque[float]] = defaultdict(deque)


class CreateRoomRequest(BaseModel):
    ttl_minutes: int = Field(default=DEFAULT_TTL_MINUTES, ge=5, le=MAX_TTL_MINUTES)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hash_code(code: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", code.encode("utf-8"), salt, PASSWORD_HASH_ROUNDS, dklen=32
    )


def _new_access_code() -> str:
    return "".join(secrets.choice("0123456789") for _ in range(8))


def _client_ip(websocket: WebSocket) -> str:
    # Do not trust X-Forwarded-For from arbitrary clients. Render terminates
    # TLS and normally exposes the connected IP as client.host.
    return websocket.client.host if websocket.client else "unknown"


def _trim_attempts(attempts: deque[float], window_seconds: float) -> None:
    now = time.monotonic()
    while attempts and now - attempts[0] > window_seconds:
        attempts.popleft()


def _check_rate_limit(bucket: deque[float], limit: int, window_seconds: float) -> bool:
    _trim_attempts(bucket, window_seconds)
    if len(bucket) >= limit:
        return False
    bucket.append(time.monotonic())
    return True


def _prune_expired_rooms() -> None:
    now = _utcnow()
    expired = [room_id for room_id, room in ROOMS.items() if room.expires_at <= now]
    for room_id in expired:
        room = ROOMS.pop(room_id, None)
        if room:
            for socket in list(room.participants.values()):
                try:
                    asyncio.create_task(socket.close(code=1008, reason="Call invitation expired"))
                except RuntimeError:
                    pass


def _configured_ice_servers() -> list[dict[str, Any]]:
    raw = os.getenv("ICE_SERVERS_JSON", "").strip()
    if raw:
        try:
            servers = json.loads(raw)
            if isinstance(servers, list) and servers and all(isinstance(x, dict) for x in servers):
                return servers
        except json.JSONDecodeError:
            pass
    return [{"urls": "stun:stun.l.google.com:19302"}]


def _allowed_origins() -> set[str]:
    return {x.strip().rstrip("/") for x in os.getenv("ALLOWED_ORIGINS", "").split(",") if x.strip()}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": APP_TITLE,
        "status": "ok",
        "note": "Signaling only; media is exchanged by browsers over WebRTC.",
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/ice")
def ice_configuration() -> dict[str, list[dict[str, Any]]]:
    """Return ICE servers; never return room secrets or stored audio."""
    return {"iceServers": _configured_ice_servers()}


@app.post("/api/rooms")
def create_room(body: CreateRoomRequest, websocket_ip: str | None = None) -> JSONResponse:
    """Create an expiring two-person room and return its access code once."""
    _prune_expired_rooms()
    # Streamlit's server-side POST cannot reliably forward a browser client IP.
    # A configurable app-level key or upstream rate limiting is recommended
    # for production deployments. The WebSocket join attempts are IP limited.
    _ = websocket_ip
    room_id = secrets.token_urlsafe(16)
    access_code = _new_access_code()
    salt = secrets.token_bytes(16)
    room = Room(
        room_id=room_id,
        salt=salt,
        code_hash=_hash_code(access_code, salt),
        expires_at=_utcnow() + timedelta(minutes=body.ttl_minutes),
    )
    ROOMS[room_id] = room
    return JSONResponse(
        {
            "room_id": room_id,
            "access_code": access_code,
            "expires_at": room.expires_at.isoformat(),
            "ttl_minutes": body.ttl_minutes,
            "max_participants": 2,
        }
    )


async def _send_if_connected(socket: WebSocket | None, message: dict[str, Any]) -> None:
    if socket is None:
        return
    try:
        await socket.send_json(message)
    except Exception:
        # The peer may have disconnected during message forwarding.
        pass


@app.websocket("/ws/{room_id}")
async def signaling(websocket: WebSocket, room_id: str) -> None:
    """Authenticate a peer, then relay WebRTC offer/answer/ICE only."""
    allowed = _allowed_origins()
    origin = websocket.headers.get("origin", "").rstrip("/")
    if allowed and origin not in allowed:
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    await websocket.accept()
    role: str | None = None
    room: Room | None = None
    client_ip = _client_ip(websocket)
    rate_bucket = AUTH_ATTEMPTS[(client_ip, room_id)]

    try:
        try:
            auth_text = await asyncio.wait_for(websocket.receive_text(), timeout=10)
            if len(auth_text.encode("utf-8")) > 2048:
                await websocket.send_json({"type": "error", "message": "Invalid join request."})
                await websocket.close(code=1008)
                return
            auth = json.loads(auth_text)
        except (asyncio.TimeoutError, json.JSONDecodeError, ValueError):
            await websocket.send_json({"type": "error", "message": "A valid access code is required."})
            await websocket.close(code=1008)
            return

        if not isinstance(auth, dict) or auth.get("type") != "auth":
            await websocket.send_json({"type": "error", "message": "Invalid join request."})
            await websocket.close(code=1008)
            return

        if not _check_rate_limit(rate_bucket, MAX_AUTH_ATTEMPTS_PER_MINUTE, 60):
            await websocket.send_json({"type": "error", "message": "Too many attempts. Wait one minute and try again."})
            await websocket.close(code=1008)
            return

        _prune_expired_rooms()
        room = ROOMS.get(room_id)
        role_value = auth.get("role")
        access_code = auth.get("access_code")
        if room is None or room.expires_at <= _utcnow():
            await websocket.send_json({"type": "error", "message": "This call invitation has expired or does not exist."})
            await websocket.close(code=1008)
            return
        if role_value not in {"host", "guest"}:
            await websocket.send_json({"type": "error", "message": "Invalid caller role."})
            await websocket.close(code=1008)
            return
        if not isinstance(access_code, str) or len(access_code) > 64:
            access_code = ""
        candidate_hash = _hash_code(access_code, room.salt)
        if not secrets.compare_digest(candidate_hash, room.code_hash):
            await websocket.send_json({"type": "error", "message": "The access code is incorrect."})
            await websocket.close(code=1008)
            return

        role = role_value
        other_role = "guest" if role == "host" else "host"
        async with room.lock:
            if role in room.participants:
                await websocket.send_json({"type": "error", "message": "Someone is already connected in this role."})
                await websocket.close(code=1008)
                return
            room.participants[role] = websocket
            other_socket = room.participants.get(other_role)

        await websocket.send_json(
            {
                "type": "ready",
                "role": role,
                "expires_at": room.expires_at.isoformat(),
                "media_encryption": "WebRTC DTLS-SRTP",
            }
        )
        if other_socket:
            await _send_if_connected(other_socket, {"type": "peer-joined", "role": role})
            await websocket.send_json({"type": "peer-joined", "role": other_role})

        allowed_signals = {"offer", "answer", "ice", "hangup"}
        while True:
            raw = await websocket.receive_text()
            if len(raw.encode("utf-8")) > MAX_SIGNAL_BYTES:
                await websocket.send_json({"type": "error", "message": "Signaling message is too large."})
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Malformed signaling message."})
                continue
            if not isinstance(message, dict) or message.get("type") not in allowed_signals:
                await websocket.send_json({"type": "error", "message": "Unsupported signaling message."})
                continue
            if message["type"] in {"offer", "answer"}:
                description = message.get("description")
                if not isinstance(description, dict) or not isinstance(description.get("sdp"), str):
                    await websocket.send_json({"type": "error", "message": "Invalid WebRTC description."})
                    continue
                if len(description["sdp"]) > MAX_SIGNAL_BYTES:
                    await websocket.send_json({"type": "error", "message": "WebRTC description is too large."})
                    continue
            elif message["type"] == "ice":
                candidate = message.get("candidate")
                if candidate is not None and not isinstance(candidate, dict):
                    await websocket.send_json({"type": "error", "message": "Invalid ICE candidate."})
                    continue

            other_socket = room.participants.get(other_role)
            await _send_if_connected(
                other_socket,
                {"type": message["type"], **{k: v for k, v in message.items() if k != "type"}},
            )
            if message["type"] == "hangup":
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        # Avoid returning exception details, codes, or state through signaling.
        try:
            await websocket.close(code=1011, reason="Signaling error")
        except Exception:
            pass
    finally:
        if room is not None and role is not None:
            async with room.lock:
                if room.participants.get(role) is websocket:
                    room.participants.pop(role, None)
                other_role = "guest" if role == "host" else "host"
                other_socket = room.participants.get(other_role)
            if other_socket:
                await _send_if_connected(other_socket, {"type": "peer-left"})
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close()
        except Exception:
            pass
