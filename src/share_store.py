"""Temporary transmission storage backed by Neon Postgres.

The database stores only the generated transmission WAV bytes and a random
share token. The secret recovery key is never stored here.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone

import psycopg


TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wavelet_transmissions (
    id UUID PRIMARY KEY,
    wav_data BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS wavelet_transmissions_expires_idx
    ON wavelet_transmissions (expires_at);
"""

TOKEN_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def ensure_schema(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(TABLE_SQL)
        conn.commit()


def cleanup_expired(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM wavelet_transmissions WHERE expires_at <= NOW()"
            )
        conn.commit()


def store_transmission(
    database_url: str,
    wav_data: bytes,
    ttl_hours: int = 168,
) -> str:
    if not wav_data:
        raise ValueError("Transmission WAV is empty.")
    if ttl_hours <= 0:
        raise ValueError("Transmission expiry must be positive.")

    token = uuid.uuid4()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(TABLE_SQL)
            cur.execute(
                """
                INSERT INTO wavelet_transmissions (id, wav_data, expires_at)
                VALUES (%s, %s, %s)
                """,
                (token, wav_data, expires_at),
            )
        conn.commit()

    return token.hex


def load_transmission(database_url: str, token: str) -> bytes:
    if not TOKEN_RE.fullmatch(token):
        raise ValueError("The receiver link contains an invalid transmission ID.")

    token_uuid = uuid.UUID(token)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT wav_data
                FROM wavelet_transmissions
                WHERE id = %s
                  AND expires_at > NOW()
                """,
                (token_uuid,),
            )
            row = cur.fetchone()

    if row is None:
        raise ValueError("This transmission link has expired or does not exist.")

    data = bytes(row[0])
    if not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
        raise ValueError("The stored transmission is not a valid WAV file.")
    return data
