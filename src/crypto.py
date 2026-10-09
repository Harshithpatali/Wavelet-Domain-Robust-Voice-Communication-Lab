"""Optional authenticated-encryption wrapper for transmission bytes.

This module is a comparison/demo mode. It uses Scrypt key derivation and
AES-256-GCM; the passphrase is never serialized with the package. Protect the
passphrase through a separate channel. Do not invent your own production
cryptographic protocol around this demo.
"""
from __future__ import annotations

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"WVE1"
SALT_LENGTH = 16
NONCE_LENGTH = 12
KEY_LENGTH = 32
MIN_PACKAGE_LENGTH = len(MAGIC) + SALT_LENGTH + NONCE_LENGTH + 16


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    if not isinstance(passphrase, str) or not passphrase.strip():
        raise ValueError("A non-empty encryption passphrase is required.")
    return Scrypt(salt=salt, length=KEY_LENGTH, n=2**14, r=8, p=1).derive(
        passphrase.encode("utf-8")
    )


def encrypt_transmission(data: bytes, passphrase: str) -> bytes:
    """Encrypt bytes into a versioned AES-GCM package with authenticated header."""
    plaintext = bytes(data)
    if not plaintext:
        raise ValueError("Cannot encrypt an empty transmission.")
    from os import urandom

    salt = urandom(SALT_LENGTH)
    nonce = urandom(NONCE_LENGTH)
    header = MAGIC + salt + nonce
    key = _derive_key(passphrase, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data=MAGIC + salt)
    return header + ciphertext


def decrypt_transmission(package: bytes, passphrase: str) -> bytes:
    """Decrypt and authenticate a package; reject wrong passphrases or tampering."""
    data = bytes(package)
    if len(data) < MIN_PACKAGE_LENGTH:
        raise ValueError("Encrypted transmission package is incomplete.")
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError("Unknown encrypted transmission format.")
    offset = len(MAGIC)
    salt = data[offset : offset + SALT_LENGTH]
    offset += SALT_LENGTH
    nonce = data[offset : offset + NONCE_LENGTH]
    offset += NONCE_LENGTH
    ciphertext = data[offset:]
    key = _derive_key(passphrase, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, associated_data=MAGIC + salt)
    except InvalidTag as exc:
        raise ValueError(
            "Authentication failed. Check the passphrase or use an unmodified package."
        ) from exc
