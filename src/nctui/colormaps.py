"""Colour maps as 256-entry RGB lookup tables."""

from __future__ import annotations

from functools import cache

import numpy as np

from ._cmapdata import CYCLIC, DIVERGING, SEQUENTIAL, TABLES

NAMES: list[str] = SEQUENTIAL + DIVERGING + CYCLIC
DEFAULT = "viridis"
DEFAULT_DIVERGING = "RdBu_r"

NAN_RGB = (90, 90, 90)


@cache
def lut(name: str) -> np.ndarray:
    """``(256, 3)`` uint8 table for ``name``."""
    return np.frombuffer(bytes.fromhex(TABLES[name]), dtype=np.uint8).reshape(256, 3)


def is_diverging(name: str) -> bool:
    return name in DIVERGING


def kind(name: str) -> str:
    if name in DIVERGING:
        return "diverging"
    if name in CYCLIC:
        return "cyclic"
    return "sequential"


@cache
def contrast_rgb(name: str) -> tuple[int, int, int]:
    """Black or white, whichever stands out more against this map overall (for coastlines)."""
    t = lut(name).astype(float) / 255
    lum = (0.2126 * t[:, 0] + 0.7152 * t[:, 1] + 0.0722 * t[:, 2]).mean()
    return (0, 0, 0) if lum > 0.5 else (255, 255, 255)


def next_name(name: str, step: int = 1) -> str:
    i = NAMES.index(name) if name in NAMES else 0
    return NAMES[(i + step) % len(NAMES)]
