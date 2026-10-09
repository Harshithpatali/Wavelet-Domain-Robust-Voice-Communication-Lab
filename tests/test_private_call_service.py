from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from call_service import app as service


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("CALL_CREATION_TOKEN", "test-only-call-creation-token-123456789")
    service.ROOMS.clear()
    service.CREATE_ATTEMPTS.clear()
    service.AUTH_ATTEMPTS.clear()
    with TestClient(service.app) as test_client:
        yield test_client
    service.ROOMS.clear()
    service.CREATE_ATTEMPTS.clear()
    service.AUTH_ATTEMPTS.clear()


def make_room(client: TestClient) -> tuple[str, str]:
    response = client.post(
        "/api/rooms",
        json={"ttl_minutes": 30},
        headers={"X-Call-Creation-Token": "test-only-call-creation-token-123456789"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    return body["room_id"], body["access_code"]


def test_health_and_ice_configuration(client: TestClient):
    assert client.get("/healthz").json() == {"status": "ok"}
    ice = client.get("/api/ice").json()["iceServers"]
    assert ice and ice[0]["urls"].startswith("stun:")


def test_room_creation_requires_server_side_creation_token(client: TestClient):
    response = client.post("/api/rooms", json={"ttl_minutes": 30})
    assert response.status_code == 503
    rejected = client.post(
        "/api/rooms",
        json={"ttl_minutes": 30},
        headers={"X-Call-Creation-Token": "wrong"},
    )
    assert rejected.status_code == 401


def test_create_room_returns_one_time_eight_digit_code(client: TestClient):
    room_id, access_code = make_room(client)
    assert len(room_id) >= 16
    assert access_code.isdigit() and len(access_code) == 8
    stored = service.ROOMS[room_id]
    assert stored.code_hash != access_code.encode()
    assert service._hash_code(access_code, stored.salt) == stored.code_hash


def test_wrong_access_code_is_rejected(client: TestClient):
    room_id, _ = make_room(client)
    with client.websocket_connect(f"/ws/{room_id}") as socket:
        socket.send_json({"type": "auth", "access_code": "00000000", "role": "host"})
        response = socket.receive_json()
        assert response["type"] == "error"
        assert "incorrect" in response["message"].lower()


def test_two_peers_can_exchange_signaling_without_media_relay(client: TestClient):
    room_id, code = make_room(client)
    with client.websocket_connect(f"/ws/{room_id}") as host:
        host.send_json({"type": "auth", "access_code": code, "role": "host"})
        assert host.receive_json()["type"] == "ready"
        with client.websocket_connect(f"/ws/{room_id}") as guest:
            guest.send_json({"type": "auth", "access_code": code, "role": "guest"})
            assert guest.receive_json()["type"] == "ready"
            assert guest.receive_json()["type"] == "peer-joined"
            assert host.receive_json()["type"] == "peer-joined"

            offer = {"type": "offer", "description": {"type": "offer", "sdp": "v=0\r\n"}}
            host.send_json(offer)
            assert guest.receive_json() == offer

            answer = {"type": "answer", "description": {"type": "answer", "sdp": "v=0\r\n"}}
            guest.send_json(answer)
            assert host.receive_json() == answer

            guest.send_json({"type": "hangup"})
            assert host.receive_json()["type"] == "hangup"


def test_room_expires_and_cannot_be_joined(client: TestClient):
    room_id, code = make_room(client)
    service.ROOMS[room_id].expires_at = service._utcnow() - timedelta(seconds=1)
    with client.websocket_connect(f"/ws/{room_id}") as socket:
        socket.send_json({"type": "auth", "access_code": code, "role": "host"})
        response = socket.receive_json()
        assert response["type"] == "error"
        assert "expired" in response["message"].lower()
