"""Temporary, expiring transmission storage backed by Neon Postgres.

Random share tokens act as bearer links. This module never stores the separate
wavelet recovery key. Access limits count receiver sessions/first loads, while
Streamlit session state can reuse a loaded WAV during that session's reruns.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone

import psycopg


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wavelet_transmissions (
    id UUID PRIMARY KEY,
    wav_data BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    max_downloads INTEGER NOT NULL DEFAULT 20,
    download_count INTEGER NOT NULL DEFAULT 0
)
"""
CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS wavelet_transmissions_expires_idx
ON wavelet_transmissions (expires_at)
"""
TOKEN_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def ensure_schema(database_url: str) -> None:
    """Create the table and safely migrate columns for existing deployments."""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(
                "ALTER TABLE wavelet_transmissions "
                "ADD COLUMN IF NOT EXISTS max_downloads INTEGER NOT NULL DEFAULT 20"
            )
            cur.execute(
                "ALTER TABLE wavelet_transmissions "
                "ADD COLUMN IF NOT EXISTS download_count INTEGER NOT NULL DEFAULT 0"
            )
            cur.execute(CREATE_INDEX_SQL)
        conn.commit()


def cleanup_expired(database_url: str) -> int:
    """Delete expired rows and return the number of rows removed."""
    ensure_schema(database_url)
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM wavelet_transmissions WHERE expires_at <= NOW()"
            )
            removed = cur.rowcount
        conn.commit()
    return int(removed)


def store_transmission(
    database_url: str,
    wav_data: bytes,
    ttl_hours: int = 168,
    max_downloads: int = 20,
) -> str:
    if not wav_data:
        raise ValueError("Transmission WAV is empty.")
    if ttl_hours <= 0:
        raise ValueError("Transmission expiry must be positive.")
    if max_downloads < 1 or max_downloads > 1000:
        raise ValueError("Receiver access limit must be between 1 and 1000.")
    if not bytes(wav_data).startswith(b"RIFF") or bytes(wav_data)[8:12] != b"WAVE":
        raise ValueError("Only valid RIFF/WAV transmissions can be shared.")

    ensure_schema(database_url)
    token = uuid.uuid4()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            # Opportunistic cleanup prevents stale shares accumulating forever.
            cur.execute("DELETE FROM wavelet_transmissions WHERE expires_at <= NOW()")
            cur.execute(
                """
                INSERT INTO wavelet_transmissions
                    (id, wav_data, expires_at, max_downloads, download_count)
                VALUES (%s, %s, %s, %s, 0)
                """,
                (token, bytes(wav_data), expires_at, int(max_downloads)),
            )
        conn.commit()
    return token.hex


def load_transmission(database_url: str, token: str) -> bytes:
    if not TOKEN_RE.fullmatch(token):
        raise ValueError("The receiver link contains an invalid transmission ID.")

    token_uuid = uuid.UUID(token)
    ensure_schema(database_url)
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT wav_data, download_count, max_downloads
                FROM wavelet_transmissions
                WHERE id = %s AND expires_at > NOW()
                FOR UPDATE
                """,
                (token_uuid,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(
                    "This transmission link has expired or does not exist."
                )
            wav_data, download_count, max_downloads = row
            if int(download_count) >= int(max_downloads):
                raise ValueError(
                    "This receiver link has reached its access limit. Ask the sender "
                    "to create a new link."
                )
            cur.execute(
                """
                UPDATE wavelet_transmissions
                SET download_count = download_count + 1
                WHERE id = %s
                """,
                (token_uuid,),
            )
        conn.commit()

    data = bytes(wav_data)
    if not data.startswith(b"RIFF") or len(data) < 12 or data[8:12] != b"WAVE":
        raise ValueError("The stored transmission is not a valid WAV file.")
    return data
