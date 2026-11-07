"""Natural Earth 110 m coastlines (public domain), drawn onto a lat/lon pixel grid."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np


@lru_cache(maxsize=1)
def load() -> tuple[np.ndarray, np.ndarray]:
    """Longitudes and latitudes of all coastline points, NaN between lines."""
    with resources.files("nctui").joinpath("data/ne_110m_coastline.npz").open("rb") as fh:
        data = np.load(fh)
        return data["lon"].astype(np.float64), data["lat"].astype(np.float64)


def _to_pixels(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Map coordinate values linearly onto pixel positions given pixel-centre coordinates."""
    n = len(centres)
    if n < 2 or centres[-1] == centres[0]:
        return np.zeros_like(values)
    return (values - centres[0]) / (centres[-1] - centres[0]) * (n - 1)


def wrap(lon: np.ndarray, lat: np.ndarray, lon_base: float, lat_base: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Coastline points brought into a plot's ranges: longitude ``lon_base`` … + 360, and,
    for a wrapped latitude axis, latitude ``lat_base`` … + 180. NaN where a line jumps."""
    lon_w = lon_base + np.mod(lon - lon_base, 360.0)
    lat_w = lat if lat_base is None else lat_base + np.mod(lat - lat_base, 180.0)
    jumps = (np.abs(np.diff(lon_w)) > 180) | (np.abs(np.diff(lat_w)) > 90)
    lon_w = lon_w.copy()
    lon_w[1:][jumps] = np.nan
    return lon_w, lat_w


def _base(centres: np.ndarray) -> float:
    lo, hi = float(np.min(centres)), float(np.max(centres))
    return lo - (hi - lo) / max(len(centres) - 1, 1) / 2


def rasterise(lon_centres: np.ndarray, lat_centres: np.ndarray, lat_wraps: bool = False) -> np.ndarray:
    """Boolean ``(len(lat_centres), len(lon_centres))`` mask of coastline pixels.

    ``lon_centres`` and ``lat_centres`` are the coordinates at the centre of each
    output pixel, in screen order (left to right, top to bottom). ``lat_wraps``: the
    latitude axis is centred, so it runs on past 90 into the south.
    """
    w, h = len(lon_centres), len(lat_centres)
    mask = np.zeros((h, w), dtype=bool)
    if w == 0 or h == 0:
        return mask
    lon, lat = load()
    # bring the coastline into the data's conventions (e.g. longitude 0..360)
    lon_w, lat_w = wrap(lon, lat, _base(lon_centres), _base(lat_centres) if lat_wraps else None)
    px = _to_pixels(lon_w, lon_centres)
    py = _to_pixels(lat_w, lat_centres)
    finite = np.isfinite(px) & np.isfinite(py)
    ok = finite[:-1] & finite[1:]
    for i in np.flatnonzero(ok):
        x0, y0, x1, y1 = px[i], py[i], px[i + 1], py[i + 1]
        # skip segments entirely outside the window
        if max(x0, x1) < -1 or min(x0, x1) > w or max(y0, y1) < -1 or min(y0, y1) > h:
            continue
        steps = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 1
        t = np.linspace(0.0, 1.0, steps + 1)
        xs = np.rint(x0 + (x1 - x0) * t).astype(int)
        ys = np.rint(y0 + (y1 - y0) * t).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        mask[ys[inside], xs[inside]] = True
    return mask
