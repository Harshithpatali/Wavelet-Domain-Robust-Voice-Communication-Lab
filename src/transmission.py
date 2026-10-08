"""Compatibility shim for older deployments.

Transmission WAV support now lives in :mod:`src.wavelet_codec`.
This module keeps older app builds/import paths working during deployment
refreshes without duplicating the codec implementation.
"""

from src.wavelet_codec import load_transmission_wav, make_transmission_wav

__all__ = ["load_transmission_wav", "make_transmission_wav"]
