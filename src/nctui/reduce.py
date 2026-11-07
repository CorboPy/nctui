"""Fit a window of data onto the screen's pixel grid.

When there are more data points than pixels along an axis, neighbouring points
are combined (mean, max, min, max |value| or nearest). When there are fewer, each
point is repeated a whole number of times, so blocks stay evenly sized.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

METHODS = ("mean", "max", "min", "maxabs", "nearest")
METHOD_LABELS = {
    "mean": "mean",
    "max": "max",
    "min": "min",
    "maxabs": "max |value|",
    "nearest": "nearest",
}


def bin_edges(n: int, m: int) -> np.ndarray:
    """Split ``n`` points into ``m`` contiguous, non-empty bins (``n >= m``)."""
    return np.floor(np.linspace(0, n, m + 1)).astype(np.int64)


def fit_length(n: int, pixels: int) -> int:
    """Pixels used for ``n`` points in ``pixels`` available: whole repeats, or all of them."""
    if n <= 0 or pixels <= 0:
        return 0
    if pixels >= n:
        return n * (pixels // n)
    return pixels


@dataclass(frozen=True)
class AxisMap:
    """How one data axis (a window of it) maps onto output pixels."""

    start: int  # first native index of the window
    n: int  # window length in data points
    out: int  # output pixels
    reverse: bool = False  # True: native index increases towards pixel 0's opposite end

    @property
    def upscale(self) -> int:
        return self.out // self.n if self.out >= self.n else 0

    @property
    def factor(self) -> float:
        """Data points per pixel (≤ 1 means every point is visible)."""
        return self.n / self.out if self.out else math.inf

    def _edges(self) -> np.ndarray:
        return bin_edges(self.n, self.out)

    def pixel_to_index(self, p: int) -> int:
        """Native index under output pixel ``p`` (the middle point of its bin)."""
        p = int(np.clip(p, 0, self.out - 1))
        if self.reverse:
            p = self.out - 1 - p
        if self.out >= self.n:
            return self.start + p // self.upscale
        e = self._edges()
        return self.start + int((e[p] + e[p + 1] - 1) // 2)

    def index_to_pixel(self, i: int) -> int | None:
        """Output pixel showing native index ``i`` (centre of its block); None if outside."""
        j = i - self.start
        if j < 0 or j >= self.n:
            return None
        if self.out >= self.n:
            k = self.upscale
            p = j * k + k // 2
        else:
            p = int(np.searchsorted(self._edges(), j, side="right") - 1)
        return self.out - 1 - p if self.reverse else p

    def pixel_span(self, i: int) -> tuple[int, int] | None:
        """First and last output pixel showing native index ``i``."""
        j = i - self.start
        if j < 0 or j >= self.n:
            return None
        if self.out >= self.n:
            k = self.upscale
            a, b = j * k, j * k + k - 1
        else:
            a = b = int(np.searchsorted(self._edges(), j, side="right") - 1)
        if self.reverse:
            a, b = self.out - 1 - b, self.out - 1 - a
        return a, b

    def pixel_indices(self) -> np.ndarray:
        """Native index under every output pixel, in pixel order."""
        return np.array([self.pixel_to_index(p) for p in range(self.out)], dtype=np.int64)


def reduce_axis(a: np.ndarray, axis: int, m: int, method: str = "mean") -> np.ndarray:
    """Combine ``a`` along ``axis`` into ``m`` bins (``m`` < length)."""
    n = a.shape[axis]
    if m >= n:
        return a
    edges = bin_edges(n, m)
    starts = edges[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "mean":
            valid = np.isfinite(a)
            total = np.add.reduceat(np.where(valid, a, 0.0), starts, axis=axis)
            count = np.add.reduceat(valid.astype(np.int64), starts, axis=axis)
            out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        elif method == "max":
            out = np.fmax.reduceat(a, starts, axis=axis)
        elif method == "min":
            out = np.fmin.reduceat(a, starts, axis=axis)
        elif method == "maxabs":
            hi = np.fmax.reduceat(a, starts, axis=axis)
            lo = np.fmin.reduceat(a, starts, axis=axis)
            out = np.where(np.abs(lo) > np.abs(hi), lo, hi)
        elif method == "nearest":
            mid = (edges[:-1] + edges[1:] - 1) // 2
            out = np.take(a, mid, axis=axis)
        else:
            raise ValueError(f"unknown method {method!r}")
    return out


def fit_axis(a: np.ndarray, axis: int, out: int, method: str = "mean") -> np.ndarray:
    n = a.shape[axis]
    if out >= n:
        k = out // n
        return np.repeat(a, k, axis=axis) if k > 1 else a
    return reduce_axis(a, axis, out, method)


def output_shape(
    nx: int,
    ny: int,
    px: int,
    py: int,
    equal: bool,
    aspect: float = 1.0,
) -> tuple[int, int]:
    """Output (width, height) in pixels for an ``ny`` × ``nx`` window.

    ``aspect`` is the window's width/height in data units, used when ``equal``.
    """
    if px <= 0 or py <= 0 or nx <= 0 or ny <= 0:
        return 0, 0
    if not equal or not np.isfinite(aspect) or aspect <= 0:
        return fit_length(nx, px), fit_length(ny, py)
    h = min(float(py), px / aspect)
    w = h * aspect
    return fit_length(nx, max(1, int(w + 1e-9))), fit_length(ny, max(1, int(h + 1e-9)))


def every_point_lengths(
    nx: int, ny: int, px: int, py: int, equal: bool, point_aspect: float = 1.0
) -> tuple[int, int]:
    """Largest window (wx, wy) that shows every point with at least one pixel each.

    ``point_aspect`` is the width/height of one grid cell in data units.
    """
    if not equal or not np.isfinite(point_aspect) or point_aspect <= 0:
        return min(nx, px), min(ny, py)
    if point_aspect >= 1:
        return max(1, min(nx, int(px / point_aspect))), min(ny, py)
    return min(nx, px), max(1, min(ny, int(py * point_aspect)))


def centred_window(n: int, length: int, centre: int) -> tuple[int, int]:
    """A ``length``-long window of ``n`` points, centred on ``centre`` and kept inside."""
    length = int(np.clip(length, 1, n))
    start = int(np.clip(centre - length // 2, 0, n - length))
    return start, start + length


def resample(
    a: np.ndarray,
    xmap: AxisMap,
    ymap: AxisMap,
    method: str = "mean",
) -> np.ndarray:
    """Fit a 2-D ``[y, x]`` window to ``(ymap.out, xmap.out)`` pixels, in screen order.

    Row 0 of the result is the top of the screen and column 0 the left.
    """
    out = fit_axis(a, 1, xmap.out, method)
    out = fit_axis(out, 0, ymap.out, method)
    if xmap.reverse:
        out = out[:, ::-1]
    if ymap.reverse:
        out = out[::-1, :]
    return out


def resample_1d(a: np.ndarray, amap: AxisMap, method: str = "mean") -> np.ndarray:
    out = fit_axis(a, 0, amap.out, method)
    return out[::-1] if amap.reverse else out
