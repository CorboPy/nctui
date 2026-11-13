"""Open netCDF files lazily and read windows of variables.

Nothing in here knows about the terminal. A :class:`Source` wraps one file, or
several files joined along their time dimension, and hands out ``float64`` numpy
arrays for the part of a variable that is being looked at. Only that part is
read from disk.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace

import netCDF4
import numpy as np

TIME, Z, Y, X, OTHER = "time", "z", "y", "x", "other"

# An index into one dimension is either a single position or a [start, stop) window.
Index = int | tuple[int, int]

# Dimensions collapsed by a statistic: (dimension, statistic) pairs.
STATS = ("mean", "min", "max", "sum")
Reduce = tuple[tuple[str, str], ...]
REDUCE_BLOCK_BYTES = 32 * 1024 * 1024  # most data read at once while reducing


class SourceError(Exception):
    """A file can't be opened or the files can't be joined."""


# --------------------------------------------------------------------------- dimensions

_Z_STANDARD_NAMES = {
    "height",
    "altitude",
    "depth",
    "air_pressure",
    "model_level_number",
    "atmosphere_hybrid_height_coordinate",
    "atmosphere_hybrid_sigma_pressure_coordinate",
    "atmosphere_sigma_coordinate",
    "atmosphere_ln_pressure_coordinate",
    "ocean_sigma_coordinate",
    "ocean_s_coordinate",
    "geopotential_height",
}
_Z_NAME = re.compile(
    r"(^|_)(levels?|lev|plev|height|heights|altitude|depth|depths|pressure|z|"
    r"model_level_number|sigma|eta|hybrid|soil)(_|$)",
    re.IGNORECASE,
)
_Y_NAMES = {"lat", "latitude", "y", "yc", "nav_lat", "rlat", "grid_latitude",
            "projection_y_coordinate", "south_north", "nlat", "j"}
_X_NAMES = {"lon", "longitude", "x", "xc", "nav_lon", "rlon", "grid_longitude",
            "projection_x_coordinate", "west_east", "nlon", "i"}
_Y_STANDARD_NAMES = {"latitude", "grid_latitude", "projection_y_coordinate"}
_X_STANDARD_NAMES = {"longitude", "grid_longitude", "projection_x_coordinate"}
_LAT_UNITS = {"degrees_north", "degree_north", "degree_n", "degrees_n", "degreen", "degreesn"}
_LON_UNITS = {"degrees_east", "degree_east", "degree_e", "degrees_e", "degreee", "degreese"}


def classify_dim(name: str, attrs: Mapping) -> str:
    """Guess what a dimension is from its coordinate variable's name and attributes."""
    axis = str(attrs.get("axis", "")).strip().upper()
    std = str(attrs.get("standard_name", "")).strip().lower()
    units = str(attrs.get("units", "")).strip().lower()
    lname = name.lower()

    if axis == "T" or std == "time" or " since " in units:
        return TIME
    if axis == "Z" or "positive" in attrs or std in _Z_STANDARD_NAMES:
        return Z
    if axis == "Y" or std in _Y_STANDARD_NAMES or units in _LAT_UNITS or lname in _Y_NAMES:
        return Y
    if axis == "X" or std in _X_STANDARD_NAMES or units in _LON_UNITS or lname in _X_NAMES:
        return X
    if lname.startswith("time") or lname in {"t", "date", "valid_time"}:
        return TIME
    if _Z_NAME.search(lname):
        return Z
    return OTHER


def _canonical_units(units: str) -> str:
    u = units.strip().lower()
    if u in _LAT_UNITS or u in _LON_UNITS or u in {"degrees", "degree", "deg"}:
        return "degrees"
    return u


@dataclass
class Dim:
    """One dimension and its coordinate."""

    name: str
    size: int
    role: str
    values: np.ndarray  # numeric coordinate, float64; the index if there is no coordinate
    units: str = ""
    long_name: str = ""
    positive: str = "up"  # "up" or "down" (for vertical coordinates)
    has_coord: bool = False
    dates: list | None = None  # decoded cftime dates for a time dimension
    lead: list[str] | None = None  # forecast lead time labels ("T+30 min")
    period: float | None = None  # set while the axis is centred (wrapped round): 360 for longitude
    label_base: float = 0.0  # labels of a centred axis read from here to here + period
    true_values: np.ndarray | None = None  # a centred latitude's rows, as in the file

    @property
    def is_lat(self) -> bool:
        return self.role == Y and _canonical_units(self.units) == "degrees"

    @property
    def is_lon(self) -> bool:
        return self.role == X and _canonical_units(self.units) == "degrees"

    @property
    def canonical_units(self) -> str:
        return _canonical_units(self.units)

    def label(self, i: int, *, with_lead: bool = True) -> str:
        """A human readable label for position ``i``."""
        if self.dates is not None:
            text = _format_date(self.dates[i], self.dates)
            if with_lead and self.lead is not None:
                text += f" ({self.lead[i]})"
            return text
        if not self.has_coord:
            return str(i)
        if self.true_values is not None:  # the north and south pole rows meet at the same value
            return format_number(float(self.true_values[i]))
        return format_number(self.wrap_value(float(self.values[i])))

    def axis_title(self) -> str:
        units = f" ({self.units})" if self.units and self.dates is None else ""
        return f"{self.name}{units}"

    def wrap_value(self, v: float) -> float:
        """A coordinate as it should read: a centred axis's values run on past the
        period (135…495 for longitude centred on 315), its labels don't (135…180, -180…135).
        A centred latitude is mirrored at the poles: past 90 it comes back down, so 100 reads 80."""
        if not self.period:
            return v
        if self.is_lat:
            return fold_latitude(v)
        return self.label_base + float(np.mod(v - self.label_base, self.period))

    def tick_label(self, v: float) -> str:
        """Label for a tick at ``v`` on a centred axis, as it reads."""
        return format_number(self.wrap_value(v))

    def nearest(self, value: float) -> int:
        """Index of the coordinate value closest to ``value``."""
        if self.size == 0:
            return 0
        diff = self.values - value
        if self.period:
            diff = np.mod(diff + self.period / 2, self.period) - self.period / 2  # 350° is -10°
        return int(np.nanargmin(np.abs(diff)))

    def find(self, text: str) -> int:
        """Index for a typed value: ``#12`` is an index, a date for time, else a number."""
        text = text.strip()
        if text.startswith("#"):
            return int(np.clip(int(text[1:]), 0, self.size - 1))
        if self.dates is not None:
            for i, d in enumerate(self.dates):
                if _format_date(d, self.dates).startswith(text) or str(d).startswith(text):
                    return i
            raise ValueError(f"no time matching {text!r}")
        return self.nearest(float(text))

    def increasing(self) -> bool:
        if self.size < 2:
            return True
        return bool(self.values[-1] >= self.values[0])

    def time_ticks(self, lo: float, hi: float, max_ticks: int) -> list[float]:
        """Tick values for a time axis between ``lo`` and ``hi``: actual time steps.

        Ticks sit on the data's own times (so they line up with the plotted columns),
        every *k*-th step where *k* makes a round interval (1 h, 6 h, 1 day...) and
        there are at most ``max_ticks``. The first tick is the roundest time on offer
        (on the hour, midnight, the 1st of the month, January).
        """
        if self.dates is None or self.size == 0:
            return []
        vals = self.values
        a, b = min(lo, hi), max(lo, hi)
        tol = 1e-9 * max(1.0, abs(a), abs(b))
        sel = np.flatnonzero((vals >= a - tol) & (vals <= b + tol))
        if len(sel) == 0:
            return []
        need = -(-len(sel) // max(1, max_ticks))  # smallest stride that fits
        stride = period = None
        if len(sel) >= 2:
            steps = np.diff(vals[sel])
            if steps[0] != 0 and np.allclose(steps, steps[0], rtol=1e-6):
                try:
                    secs = abs((self.dates[sel[1]] - self.dates[sel[0]]).total_seconds())
                except (TypeError, AttributeError):
                    secs = 0.0
                for nice in _NICE_SECONDS if secs > 0 else ():
                    k = nice / secs
                    if k >= need - 1e-9 and abs(k - round(k)) < 1e-6:
                        stride, period = int(round(k)), nice
                        break
        if stride is None:
            stride = next((k for k in _NICE_STRIDES if k >= need), need)
        first = max(range(min(stride, len(sel))), key=lambda o: (_roundness(self.dates[sel[o]], period), -o))
        return [float(vals[i]) for i in sel[first::stride]]


_NICE_SECONDS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 2 * 3600, 3 * 3600, 6 * 3600,
                 12 * 3600, 86400, 2 * 86400, 5 * 86400, 10 * 86400)
_NICE_STRIDES = (1, 2, 3, 4, 5, 6, 10, 12, 20, 24, 50, 60, 100, 120, 200, 240, 500, 1000)


def fold_latitude(u: float) -> float:
    """The latitude shown at ``u`` on a latitude axis mirrored at the poles.

    Going on past the north pole the map comes back down (u = 100 is latitude 80),
    and past the south pole back up (u = -100 is latitude -80).
    """
    u = float(np.mod(u + 180.0, 360.0) - 180.0)
    if u > 90.0:
        return 180.0 - u
    if u < -90.0:
        return -180.0 - u
    return u


def _mirror_layout(stored: np.ndarray, centre: float) -> tuple[np.ndarray, np.ndarray]:
    """A latitude axis mirrored at both poles, one full turn (360°) centred on ``centre``.

    Every row of the file appears twice: as itself (u = latitude) and mirrored
    (u = 180 - latitude past the north pole, or -180 - latitude past the south). A
    row exactly on a pole is its own mirror, so it appears once. Returns the display
    coordinate ``u`` and the file row for each, in the file's direction.
    """
    n = len(stored)
    u = np.concatenate([stored, 180.0 - stored])
    rows = np.concatenate([np.arange(n), np.arange(n)])
    keep = np.ones(2 * n, dtype=bool)
    keep[n:] = ~np.isclose(np.abs(stored), 90.0)
    u, rows = u[keep], rows[keep]
    u = centre + np.mod(u - centre + 180.0, 360.0) - 180.0
    order = np.argsort(u, kind="stable")
    if stored[-1] < stored[0]:
        order = order[::-1]
    return u[order].astype(np.float64), rows[order].astype(np.int64)


def _pole_to_pole(lat: np.ndarray) -> bool:
    """Does a latitude grid reach both poles (to within about half a grid cell)?"""
    if len(lat) < 2:
        return False
    s = np.sort(lat)
    return bool(s[-1] + 0.6 * (s[-1] - s[-2]) >= 90.0 and s[0] - 0.6 * (s[1] - s[0]) <= -90.0)


def same_units(a: Dim, b: Dim) -> bool:
    """Do two coordinates share units, so one unit along each is the same distance?"""
    return a.has_coord and b.has_coord and bool(a.canonical_units) and a.canonical_units == b.canonical_units


def _roundness(d, period: float | None) -> int:
    """How round a date is as a place to start ticks every ``period`` seconds."""
    try:
        sod = d.hour * 3600 + d.minute * 60 + d.second
        score = 0
        if period and period <= 86400 and sod % period == 0:
            score += 8
        if sod == 0:
            score += 4
        if d.day == 1:
            score += 2
        if d.month == 1:
            score += 1
        return score
    except AttributeError:
        return 0


def _format_date(d, all_dates: list) -> str:
    try:
        if all(getattr(x, "second", 0) == 0 for x in all_dates[:50]):
            if all(getattr(x, "hour", 0) == 0 and getattr(x, "minute", 0) == 0 for x in all_dates[:50]):
                return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
            return f"{d.year:04d}-{d.month:02d}-{d.day:02d} {d.hour:02d}:{d.minute:02d}"
        return f"{d.year:04d}-{d.month:02d}-{d.day:02d} {d.hour:02d}:{d.minute:02d}:{d.second:02d}"
    except AttributeError:
        return str(d)


def format_number(v: float, digits: int = 5) -> str:
    """Compact number formatting for labels: 0.5, 120.5, 1.25e-05, 6.4e+05."""
    if v is None or not np.isfinite(v):
        return "nan"
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-3 <= a < 1e5:
        text = f"{v:.{digits}g}"
    else:
        text = f"{v:.{max(digits - 2, 1)}e}"
        mant, exp = text.split("e")
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")
        text = f"{mant}e{int(exp):+03d}"
    return text


def format_lead(seconds: float) -> str:
    """Forecast lead time as T+30 min, T+1 h 30 min, T+2 d 3 h."""
    if not np.isfinite(seconds):
        return "T+?"
    s = int(round(seconds))
    sign = "+" if s >= 0 else "-"
    s = abs(s)
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d} d")
    if h:
        parts.append(f"{h} h")
    if m:
        parts.append(f"{m} min")
    if sec and not d:
        parts.append(f"{sec} s")
    return f"T{sign}" + (" ".join(parts) if parts else "0")


def format_leads(seconds: Iterable[float]) -> list[str]:
    """Lead times for a whole time axis, all with the same fields so they line up.

    Every label shows the units from the largest used anywhere down to the smallest,
    with hours, minutes and seconds as two digits after a larger unit: T+7 d 02 h 30 min
    and T+7 d 03 h 00 min, not T+7 d 2 h 30 min and T+7 d 3 h. That keeps an
    animation's title from jumping about as the label changes length.
    """
    vals = [int(round(v)) if np.isfinite(v) else None for v in seconds]
    parts = [(abs(v) // 86400, abs(v) % 86400 // 3600, abs(v) % 3600 // 60, abs(v) % 60) for v in vals
             if v is not None]
    if not parts:
        return ["T+?"] * len(vals)
    used = [any(p[k] for p in parts) for k in range(4)]
    if not any(used):
        return ["T+0" if v is not None else "T+?" for v in vals]
    first = used.index(True)
    last = 3 - used[::-1].index(True)
    names = ("d", "h", "min", "s")
    out = []
    for v in vals:
        if v is None:
            out.append("T+?")
            continue
        a = abs(v)
        p = (a // 86400, a % 86400 // 3600, a % 3600 // 60, a % 60)
        fields = [f"{p[k]:02d} {names[k]}" if k > first else f"{p[k]} {names[k]}" for k in range(first, last + 1)]
        out.append(("T+" if v >= 0 else "T-") + " ".join(fields))
    return out


# --------------------------------------------------------------------------- variables


@dataclass
class Var:
    """One variable's metadata."""

    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    attrs: dict = field(default_factory=dict)
    plottable: bool = True

    @property
    def units(self) -> str:
        return str(self.attrs.get("units", ""))

    @property
    def long_name(self) -> str:
        return str(self.attrs.get("long_name", "") or self.attrs.get("standard_name", ""))

    @property
    def standard_name(self) -> str:
        return str(self.attrs.get("standard_name", ""))

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def matches(self, text: str) -> bool:
        t = text.lower()
        return t in self.name.lower() or t in self.long_name.lower() or t in self.standard_name.lower()


# --------------------------------------------------------------------------- source


class _LRU:
    """Least-recently-used cache with a byte budget."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        return None

    def put(self, key, value: np.ndarray) -> None:
        size = int(getattr(value, "nbytes", 0))
        if size > self.max_bytes:
            return
        with self._lock:
            if key in self._items:
                self._bytes -= int(self._items.pop(key).nbytes)
            self._items[key] = value
            self._bytes += size
            while self._bytes > self.max_bytes and self._items:
                _, old = self._items.popitem(last=False)
                self._bytes -= int(old.nbytes)


class Source:
    """One netCDF file, or several joined along time, opened lazily."""

    def __init__(self, paths: str | os.PathLike | Iterable[str | os.PathLike], cache_mb: int = 256):
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        self.paths = [str(p) for p in paths]
        if not self.paths:
            raise SourceError("no files given")
        self.warnings: list[str] = []
        self._read_lock = threading.Lock()
        # centred (wrapped) axes: display index j shows stored index (j + roll) % size
        self._roll: dict[str, int] = {}
        # mirrored (centred) latitudes: display index j shows the file's row _mirror[d][j]
        self._mirror: dict[str, np.ndarray] = {}
        self._centres: dict[str, float] = {}
        self._stored_values: dict[str, np.ndarray] = {}
        self._cache = _LRU(cache_mb * 1024 * 1024)
        # reduced fields are kept apart, so the slices read to make the next one don't push them out
        self._reduced = _LRU(cache_mb * 1024 * 1024)

        datasets = [self._open(p) for p in self.paths]
        self.global_attrs = _attrs(datasets[0])
        self.time_dim = self._find_time_dim(datasets[0])
        self._common_vars: set[str] | None = None
        if len(datasets) > 1:
            datasets, self.paths = self._sort_and_check(datasets, self.paths)
        self._datasets = datasets
        self._offsets = np.cumsum([0] + [
            (len(ds.dimensions[self.time_dim]) if self.time_dim in ds.dimensions else 0) for ds in datasets
        ])

        self.dims: dict[str, Dim] = {}
        self.vars: dict[str, Var] = {}
        self._build_dims()
        self._build_vars()

    # ------------------------------------------------------------------ opening

    @staticmethod
    def _open(path: str) -> netCDF4.Dataset:
        if not os.path.exists(path):
            raise SourceError(f"{path}: no such file")
        try:
            ds = netCDF4.Dataset(path, "r")
        except Exception as exc:  # noqa: BLE001 - report any reason the file won't open
            raise SourceError(f"{path}: can't open ({exc})") from exc
        ds.set_auto_maskandscale(True)
        ds.set_always_mask(False)
        return ds

    @staticmethod
    def _find_time_dim(ds: netCDF4.Dataset) -> str | None:
        for name in ds.dimensions:
            attrs = _attrs(ds.variables[name]) if name in ds.variables else {}
            if classify_dim(str(name), attrs) == TIME:
                return str(name)
        unlimited = [n for n, d in ds.dimensions.items() if d.isunlimited()]
        return unlimited[0] if unlimited else None

    def _sort_and_check(self, datasets: list[netCDF4.Dataset], paths: list[str]):
        t = self.time_dim
        if t is None:
            raise SourceError(
                f"can't join {len(paths)} files: {os.path.basename(paths[0])} has no time dimension"
            )
        first = datasets[0]
        for ds, p in zip(datasets[1:], paths[1:], strict=True):
            if t not in ds.dimensions:
                raise SourceError(f"can't join files: {p} has no '{t}' dimension")
            for dim, d0 in first.dimensions.items():
                if dim == t:
                    continue
                if dim not in ds.dimensions:
                    raise SourceError(f"can't join files: {p} has no '{dim}' dimension")
                if len(ds.dimensions[dim]) != len(d0):
                    raise SourceError(
                        f"can't join files: '{dim}' is {len(ds.dimensions[dim])} long in {p} "
                        f"but {len(d0)} in {paths[0]}"
                    )

        def first_time(ds: netCDF4.Dataset):
            if t not in ds.variables or len(ds.dimensions[t]) == 0:
                return (0.0,)
            var = ds.variables[t]
            dates = _decode_times(_values(var[0:1]), _attrs(var))
            if dates is not None:
                d = dates[0]
                try:
                    return (d.year, d.month, d.day, d.hour, d.minute, d.second)
                except AttributeError:
                    pass
            return (float(_values(var[0:1])[0]),)

        order = sorted(range(len(datasets)), key=lambda i: first_time(datasets[i]))
        datasets = [datasets[i] for i in order]
        paths = [paths[i] for i in order]

        common = set(datasets[0].variables)
        for ds, p in zip(datasets[1:], paths[1:], strict=True):
            missing = common - set(ds.variables)
            if missing:
                self.warnings.append(
                    f"{os.path.basename(p)} lacks {', '.join(sorted(missing))}; "
                    "those variables are left out"
                )
            common &= set(ds.variables)
        self._common_vars = common
        return datasets, paths

    # ------------------------------------------------------------------ metadata

    def _build_dims(self) -> None:
        ds0 = self._datasets[0]
        for name, d in ds0.dimensions.items():
            name = str(name)
            size = len(d)
            var = ds0.variables.get(name)
            attrs = _attrs(var) if var is not None else {}
            role = classify_dim(name, attrs)
            if name == self.time_dim:
                role = TIME
                size = int(self._offsets[-1])
            has_coord = var is not None and var.dimensions == (name,)
            dim = Dim(
                name=name,
                size=int(size),
                role=role,
                values=np.arange(size, dtype=np.float64),
                units=str(attrs.get("units", "")),
                long_name=str(attrs.get("long_name", "") or attrs.get("standard_name", "")),
                positive=str(attrs.get("positive", "up")).lower(),
                has_coord=has_coord,
            )
            if has_coord:
                try:
                    if name == self.time_dim:
                        self._fill_time(dim)
                    else:
                        vals = _values(var[:])
                        if vals.shape == (dim.size,):
                            dim.values = vals
                        else:
                            dim.has_coord = False
                except (TypeError, ValueError):
                    dim.has_coord = False
            if role == Z and "positive" not in attrs:
                std = str(attrs.get("standard_name", ""))
                if std == "air_pressure" or dim.units.lower() in {"pa", "hpa", "mb", "mbar", "bar"}:
                    dim.positive = "down"
            self.dims[name] = dim

    def _fill_time(self, dim: Dim) -> None:
        t = dim.name
        raw_parts, dates = [], []
        decoded = True
        for ds in self._datasets:
            if t not in ds.variables:
                continue
            var = ds.variables[t]
            raw = _values(var[:])
            raw_parts.append(raw)
            d = _decode_times(raw, _attrs(var))
            if d is None:
                decoded = False
            else:
                dates.extend(d)
        raw = np.concatenate(raw_parts) if raw_parts else np.arange(dim.size, dtype=np.float64)
        if decoded and dates and len(dates) == dim.size:
            dim.dates = dates
            try:
                import cftime

                attrs = _attrs(self._datasets[0].variables[t])
                units = attrs.get("units")
                cal = attrs.get("calendar", "standard")
                dim.values = np.asarray(cftime.date2num(dates, units, cal), dtype=np.float64)
            except Exception:  # noqa: BLE001 - fall back to raw values
                dim.values = raw
        else:
            dim.values = raw
        # forecast lead time, if the file says
        lead_parts = []
        for ds in self._datasets:
            fp = ds.variables.get("forecast_period")
            if fp is not None and fp.dimensions == (t,):
                vals = _values(fp[:])
                unit = str(_attrs(fp).get("units", "seconds")).split()[0] if _attrs(fp).get("units") else "seconds"
                scale = {"seconds": 1, "second": 1, "s": 1, "minutes": 60, "hours": 3600, "hour": 3600,
                         "h": 3600, "days": 86400, "day": 86400}.get(unit, 1)
                lead_parts.append(vals * scale)
        if lead_parts and sum(len(p) for p in lead_parts) == dim.size:
            dim.lead = format_leads(np.concatenate(lead_parts))

    def _build_vars(self) -> None:
        ds0 = self._datasets[0]
        bounds, aux = set(), set()
        for var in ds0.variables.values():
            attrs = _attrs(var)
            if "bounds" in attrs:
                bounds.add(str(attrs["bounds"]))
            if "coordinates" in attrs:
                aux.update(str(attrs["coordinates"]).split())
        for name, var in ds0.variables.items():
            name = str(name)
            if self._common_vars is not None and name not in self._common_vars:
                continue
            dims = tuple(str(d) for d in var.dimensions)
            if dims == (name,):
                continue  # a dimension's coordinate variable
            shape = tuple(self.dims[d].size for d in dims)
            dtype = var.dtype
            numeric = isinstance(dtype, np.dtype) and (
                np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)
            )
            plottable = (
                numeric
                and name not in bounds
                and name not in aux
                and not any(d.lower() in {"bnds", "bounds", "nv", "nbnd", "nb2", "vertices"} for d in dims)
            )
            self.vars[name] = Var(
                name=name,
                dims=dims,
                shape=shape,
                dtype=str(dtype),
                attrs=_attrs(var),
                plottable=plottable,
            )

    # ------------------------------------------------------------------ description

    @property
    def title(self) -> str:
        if len(self.paths) == 1:
            return os.path.basename(self.paths[0])
        return f"{os.path.basename(self.paths[0])} + {len(self.paths) - 1} more (joined on {self.time_dim})"

    @property
    def nbytes_on_disk(self) -> int:
        return sum(os.path.getsize(p) for p in self.paths)

    def plottable_vars(self) -> list[Var]:
        return [v for v in self.vars.values() if v.plottable]

    def file_attrs(self) -> dict:
        return self.global_attrs

    # ------------------------------------------------------------------ reading

    def read(self, var: str, index: Mapping[str, Index], stored: bool = False) -> np.ndarray:
        """Read part of ``var``.

        ``index`` maps dimension names to an int (that dimension is dropped from
        the result) or a ``(start, stop)`` window. Dimensions not mentioned are
        read whole. The result is float64 with NaN where data is missing, and its
        axes keep the variable's dimension order. With ``stored``, ``index`` counts
        along the file's axes, whatever is centred.
        """
        v = self.vars[var]
        norm: dict[str, Index] = {}
        for d in v.dims:
            ix = index.get(d)
            size = self.file_size(d) if stored else self.dims[d].size
            if ix is None:
                norm[d] = (0, size)
            elif isinstance(ix, tuple):
                a, b = max(0, int(ix[0])), min(size, int(ix[1]))
                norm[d] = (a, max(a, b))
            else:
                norm[d] = int(np.clip(int(ix), 0, size - 1))
        rolls = {d: 0 if stored else self._roll.get(d, 0) for d in v.dims}
        mirrored = [] if stored else [d for d in v.dims if d in self._mirror]
        key = (var, tuple(norm[d] for d in v.dims), tuple(rolls.values()), tuple(self._centres[d] for d in mirrored))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if mirrored:
            arr = self._read_mirrored(v, norm, rolls, mirrored[0])
        else:
            arr = self._read_rolled(v, norm, rolls) if any(rolls.values()) else self._read_uncached(v, norm)
        arr.flags.writeable = False
        self._cache.put(key, arr)
        return arr

    # ------------------------------------------------------------------ centring

    def period(self, dim: str) -> float | None:
        """Length of one period if ``dim`` can be wrapped round, else None.

        Longitude can when the grid goes all the way round, and latitude when it
        reaches both poles (it is mirrored at each pole, so one turn is 360°).
        The x and y of a Cartesian grid are taken to be periodic (as model domains
        usually are), as long as they are evenly spaced. Time and levels can't.
        """
        d = self.dims[dim]
        if d.role not in (X, Y) or not d.has_coord or d.dates is not None or d.size < 3:
            return None
        vals = self._stored_values.get(dim, d.values)
        steps = np.diff(vals)
        if not (np.all(steps > 0) or np.all(steps < 0)):
            return None
        if d.is_lat:
            return 360.0 if _pole_to_pole(vals) else None
        step = float(np.median(np.abs(steps)))
        if not np.allclose(np.abs(steps), step, rtol=1e-3):
            return None
        if d.is_lon:
            return 360.0 if abs(step * d.size - 360.0) <= 0.51 * step else None
        return step * d.size

    def centre(self, dim: str) -> float | None:
        """The coordinate value ``dim`` is centred on, if it has been."""
        return self._centres.get(dim)

    def roll(self, dim: str) -> int:
        return self._roll.get(dim, 0)

    def file_size(self, dim: str) -> int:
        """Length of ``dim`` in the file (a centred latitude shows each row twice)."""
        return len(self._stored_values.get(dim, self.dims[dim].values))

    def stored_index(self, dim: str, j: int) -> int:
        """The file's index shown at display index ``j``."""
        rows = self._mirror.get(dim)
        if rows is not None:
            return int(rows[j])
        return (j + self._roll.get(dim, 0)) % self.dims[dim].size

    def display_index(self, dim: str, i: int) -> int:
        """Where the file's index ``i`` is shown (for a mirrored latitude, the unmirrored copy)."""
        rows = self._mirror.get(dim)
        if rows is not None:
            d = self.dims[dim]
            hits = np.flatnonzero(rows == i)
            true = self._stored_values[dim][i]
            return int(min(hits, key=lambda j: abs(d.values[j] - true)))
        return (i - self._roll.get(dim, 0)) % self.dims[dim].size

    def set_centre(self, dim: str, value: float | None) -> None:
        """Wrap ``dim`` round so ``value`` is in the middle (None: back to the file's order).

        From then on ``dim``'s values and every read are in the wrapped order: for
        longitude centred on 0 the axis runs -180 … 180. Latitude is mirrored at the
        poles and shows one full turn: centred on 90 it runs from the south pole up to
        the north pole in the middle and back down to the south pole, with every row
        twice. Positions and windows given to :meth:`read` count along the wrapped axis.
        """
        d = self.dims[dim]
        stored = self._stored_values.setdefault(dim, d.values.copy())
        if value is None:
            self._centres.pop(dim, None)
            self._roll.pop(dim, None)
            self._mirror.pop(dim, None)
            self.dims[dim] = replace(d, size=len(stored), values=stored, period=None, true_values=None)
            return
        period = self.period(dim)
        if period is None:
            raise ValueError(f"{dim} doesn't wrap round" + (" (it doesn't reach both poles)" if d.is_lat else ""))
        if d.is_lat:
            if not -90.0 <= value <= 90.0:
                raise ValueError("a latitude is between -90 and 90")
            u, rows = _mirror_layout(stored, float(value))
            self._centres[dim] = float(value)
            self._mirror[dim] = rows
            self.dims[dim] = replace(d, size=len(u), values=u, period=period, true_values=stored[rows])
            return
        wrapped = value + np.mod(stored - value + period / 2, period) - period / 2
        shift = int(np.argmin(wrapped) if stored[-1] > stored[0] else np.argmax(wrapped))
        self._centres[dim] = float(value)
        if shift:
            self._roll[dim] = shift
        else:
            self._roll.pop(dim, None)
        step = period / d.size
        # longitude reads -180…180 and latitude -90…90; Cartesian x/y stay inside the domain
        base = -180.0 if d.is_lon else -90.0 if d.is_lat else float(np.min(stored)) - step / 2
        self.dims[dim] = replace(d, values=np.roll(wrapped, -shift), period=period, label_base=base,
                                 true_values=None)

    def _read_mirrored(self, v: Var, norm: dict[str, Index], rolls: dict[str, int], lat: str) -> np.ndarray:
        """Read with a latitude mirrored at the poles: runs of rows going up or down the file."""
        rows = self._mirror[lat]

        def read(norm: dict[str, Index]) -> np.ndarray:
            return self._read_rolled(v, norm, rolls) if any(rolls.values()) else self._read_uncached(v, norm)

        ix = norm[lat]
        if isinstance(ix, int):
            return read({**norm, lat: int(rows[ix])})
        a, b = ix
        if b <= a:
            return read({**norm, lat: (0, 0)})
        axis = [d for d in v.dims if not isinstance(norm[d], int)].index(lat)
        parts = []
        j = a
        while j < b:
            k, step = j + 1, 0  # a run of rows one file row apart, all one way
            while k < b:
                s = int(rows[k] - rows[k - 1])
                if abs(s) != 1 or (step and s != step):
                    break
                step, k = s, k + 1
            lo, hi = int(min(rows[j], rows[k - 1])), int(max(rows[j], rows[k - 1])) + 1
            part = read({**norm, lat: (lo, hi)})
            parts.append(np.flip(part, axis=axis) if step < 0 else part)
            j = k
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=axis)

    def _read_rolled(self, v: Var, norm: dict[str, Index], rolls: dict[str, int]) -> np.ndarray:
        """Read with some dimensions wrapped round: a window may come in two pieces."""
        norm = dict(norm)
        for d, shift in rolls.items():
            if shift and isinstance(norm[d], int):
                norm[d] = (norm[d] + shift) % self.dims[d].size
        todo = [d for d, shift in rolls.items() if shift and isinstance(norm[d], tuple)]
        kept = [d for d in v.dims if not isinstance(norm[d], int)]

        def pieces(norm: dict[str, Index], todo: list[str]) -> np.ndarray:
            if not todo:
                return self._read_uncached(v, norm)
            d, rest = todo[0], todo[1:]
            n, shift = self.dims[d].size, rolls[d]
            a, b = norm[d]
            a, b = a + shift, b + shift
            if a >= n:
                a, b = a - n, b - n
            if b <= n:
                return pieces({**norm, d: (a, b)}, rest)
            first = pieces({**norm, d: (a, n)}, rest)
            second = pieces({**norm, d: (0, b - n)}, rest)
            return np.concatenate([first, second], axis=kept.index(d))

        return pieces(norm, todo)

    def _read_uncached(self, v: Var, norm: dict[str, Index]) -> np.ndarray:
        t = self.time_dim
        if len(self._datasets) == 1 or t not in v.dims:
            return self._slice(0, v, norm)
        ix = norm[t]
        if isinstance(ix, int):
            f = int(np.searchsorted(self._offsets, ix, side="right") - 1)
            local = dict(norm)
            local[t] = ix - int(self._offsets[f])
            return self._slice(f, v, local)
        a, b = ix
        pieces = []
        for f in range(len(self._datasets)):
            lo, hi = int(self._offsets[f]), int(self._offsets[f + 1])
            s, e = max(a, lo), min(b, hi)
            if s < e:
                local = dict(norm)
                local[t] = (s - lo, e - lo)
                pieces.append(self._slice(f, v, local))
        kept = [d for d in v.dims if not isinstance(norm[d], int)]
        if not pieces:
            return np.full([norm[d][1] - norm[d][0] for d in kept], np.nan)
        return np.concatenate(pieces, axis=kept.index(t))

    def _slice(self, file_index: int, v: Var, norm: dict[str, Index]) -> np.ndarray:
        key = tuple(ix if isinstance(ix, int) else slice(ix[0], ix[1]) for ix in (norm[d] for d in v.dims))
        with self._read_lock:
            raw = self._datasets[file_index].variables[v.name][key] if key else \
                self._datasets[file_index].variables[v.name].getValue()
        return _values(raw)

    # ------------------------------------------------------------------ reducing

    def reduce_read(
        self,
        var: str,
        index: Mapping[str, Index],
        reduce: Reduce,
        progress: Callable[[float], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> np.ndarray | None:
        """Read part of ``var`` with the dimensions in ``reduce`` collapsed by a statistic.

        ``reduce`` pairs dimensions with ``mean``, ``min``, ``max`` or ``sum``; ``index``
        is as in :meth:`read` for the others. Missing values are ignored, and where
        every value is missing so is the result. A reduced dimension is collapsed over
        its whole length in the file, each point once, however it is centred or zoomed.
        Dimensions that share a statistic collapse together: a mean over latitude and
        longitude is one mean of every valid point, weighted by cos(latitude) when a
        latitude in degrees is collapsed. Different statistics apply from the variable's
        last dimension to its first (space before time).

        The data is read a block at a time (see :data:`REDUCE_BLOCK_BYTES`) through the
        slice cache, tiling the dimensions kept as well as those collapsed last; only
        dimensions collapsed first, under a different statistic, are read whole. The
        result is kept for the whole of the windowed dimensions, so zooming, scrolling
        and centring afterwards don't read again. Returns None if cancelled.
        """
        if not reduce:
            return self.read(var, index)
        v = self.vars[var]
        key, stats, fixed, shown = self._reduce_plan(v, index, reduce)
        full = self._reduced.get(key)
        if full is None:
            full = self._reduce(v, stats, fixed, progress, cancelled)
            if full is None:
                return None
            full = np.asarray(full, dtype=np.float64)
            full.flags.writeable = False
            self._reduced.put(key, full)
        out = full
        for axis, (d, (a, b)) in enumerate(shown.items()):
            out = out[(slice(None),) * axis + (self._file_rows(d, a, b),)]
        out.flags.writeable = False
        return out

    def reduce_cached(self, var: str, index: Mapping[str, Index], reduce: Reduce) -> bool:
        """Would :meth:`reduce_read` return at once?"""
        return not reduce or self._reduced.get(self.reduce_key(var, index, reduce)) is not None

    def reduce_key(self, var: str, index: Mapping[str, Index], reduce: Reduce) -> tuple:
        """What a reduced field depends on: not the windows, nor how anything is centred."""
        return self._reduce_plan(self.vars[var], index, reduce)[0]

    def reduce_weighting(self, var: str, reduce: Reduce) -> str:
        """How the means in ``reduce`` are weighted, to tell the user ("" if there are none)."""
        notes = ["cos(lat)-weighted mean" if any(self.dims[d].is_lat for d in dims) else "unweighted mean"
                 for stat, dims in self._reduce_groups(self.vars[var], dict(reduce)) if stat == "mean"]
        return ", ".join(dict.fromkeys(notes))

    def _reduce_plan(self, v: Var, index: Mapping[str, Index], reduce: Reduce):
        """Cache key, statistic per reduced dimension, file position of each dimension
        held at one point, and the window shown of each of the rest."""
        stats = dict(reduce)
        for d, stat in stats.items():
            if d not in v.dims:
                raise ValueError(f"{v.name} has no dimension {d}")
            if stat not in STATS:
                raise ValueError(f"unknown statistic {stat!r}: choose from {', '.join(STATS)}")
        fixed: dict[str, int] = {}
        shown: dict[str, tuple[int, int]] = {}
        for d in v.dims:
            if d in stats:
                continue
            ix = index.get(d)
            size = self.dims[d].size
            if ix is None:
                shown[d] = (0, size)
            elif isinstance(ix, tuple):
                a, b = max(0, int(ix[0])), min(size, int(ix[1]))
                shown[d] = (a, max(a, b))
            else:
                fixed[d] = self.stored_index(d, int(np.clip(int(ix), 0, size - 1)))
        key = (v.name, tuple((d, stats[d]) for d in v.dims if d in stats),
               tuple(fixed.get(d) for d in v.dims if d not in stats))
        return key, stats, fixed, shown

    @staticmethod
    def _reduce_groups(v: Var, stats: Mapping[str, str]) -> list[tuple[str, list[str]]]:
        """The order to collapse in: last dimension first, neighbours with one statistic together."""
        groups: list[tuple[str, list[str]]] = []
        for d in reversed(v.dims):
            if d not in stats:
                continue
            if groups and groups[-1][0] == stats[d]:
                groups[-1][1].insert(0, d)
            else:
                groups.append((stats[d], [d]))
        return groups

    def _file_rows(self, dim: str, a: int, b: int) -> slice | np.ndarray:
        """The file's indices shown at display indices ``a`` to ``b`` of ``dim``."""
        rows = self._mirror.get(dim)
        if rows is not None:
            return rows[a:b]
        shift = self._roll.get(dim, 0)
        return (np.arange(a, b) + shift) % self.dims[dim].size if shift else slice(a, b)

    def _reduce(self, v: Var, stats: Mapping[str, str], fixed: Mapping[str, int], progress, cancelled):
        """Collapse the dimensions in ``stats``, over the whole of the file's axes."""
        groups = self._reduce_groups(v, stats)
        last_stat, last = groups[-1]  # the group with the outermost dimension: summed up block by block
        axes = [d for d in v.dims if d not in fixed]
        whole = {d: (0, self.file_size(d)) for d in axes}
        nbytes = 8.0 * math.prod(self.file_size(d) for d in axes)  # Python ints: no overflow
        # tile the dimensions kept in the result and those of the last group, outermost first,
        # until a block is small enough (an earlier group has to be read whole to be collapsed)
        kept = [d for d in axes if d not in stats]
        tiled = [d for d in axes if d in kept or d in last]
        spans = {d: [whole[d]] for d in tiled}
        for d in tiled:
            n = self.file_size(d)
            if nbytes <= REDUCE_BLOCK_BYTES or n == 0:
                break
            step = max(1, int(REDUCE_BLOCK_BYTES // (nbytes / n)))
            spans[d] = [(a, min(n, a + step)) for a in range(0, n, step)]
            nbytes = nbytes / n * step
        blocks = list(itertools.product(*(spans[d] for d in tiled)))
        shape = [self.file_size(d) for d in kept]
        extreme = last_stat in ("min", "max")
        acc = (np.full(shape, np.nan), None) if extreme else (np.zeros(shape), np.zeros(shape))

        def weights(dims: list[str], stat: str, names: list[str], norm: Mapping[str, Index]) -> np.ndarray | None:
            lat = next((d for d in dims if self.dims[d].is_lat), None) if stat == "mean" else None
            if lat is None:
                return None
            ix = norm[lat]
            a, b = (ix, ix + 1) if isinstance(ix, int) else ix
            w = np.clip(np.cos(np.deg2rad(self._stored_values.get(lat, self.dims[lat].values)[a:b])), 0.0, None)
            return w.reshape([-1 if n == lat else 1 for n in names])

        for k, block in enumerate(blocks):
            if cancelled is not None and cancelled():
                return None
            norm: dict[str, Index] = {**fixed, **whole}
            for d, (a, b) in zip(tiled, block, strict=True):
                # one step of a reduced dimension: the slice a map of it reads too
                norm[d] = a if b - a == 1 and d in last else (a, b)
            data = self.read(v.name, norm, stored=True)
            for i, d in enumerate(axes):
                if isinstance(norm[d], int):
                    data = np.expand_dims(data, i)  # keep its axis, to collapse below
            names = list(axes)
            for stat, dims in groups[:-1]:
                part = _partial(data, tuple(names.index(d) for d in dims), stat, weights(dims, stat, names, norm))
                data = _finish(part, stat)
                names = [n for n in names if n not in dims]
            part = _partial(data, tuple(names.index(d) for d in last), last_stat,
                            weights(last, last_stat, names, norm))
            here = tuple(slice(*norm[d]) for d in kept)  # this block's part of the result
            merged = _merge((acc[0][here], None if extreme else acc[1][here]), part, last_stat)
            acc[0][here] = merged[0]
            if not extreme:
                acc[1][here] = merged[1]
            if progress is not None:
                progress((k + 1) / len(blocks))
        return _finish(acc, last_stat)

    def value_range(
        self,
        var: str,
        index: Mapping[str, Index] | None = None,
        sweep: str | None = None,
        progress: Callable[[float], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        reduce: Reduce = (),
    ) -> tuple[float, float] | None:
        """Minimum and maximum of ``var`` over every position along ``sweep``.

        The other dimensions are indexed by ``index`` (a position, or a window as
        in :meth:`read`), or collapsed as ``reduce`` says. With no ``sweep``, the
        whole variable (or, with ``reduce``, the reduced field ``index`` selects).
        Reads one slice at a time; returns None if cancelled or there is no finite data.
        """
        v = self.vars[var]
        if v.ndim == 0:
            val = float(np.asarray(self.read(var, {})).reshape(-1)[0])
            return (val, val)
        if sweep is None and reduce:  # nothing to step through: the range of the reduced field itself
            field = self.reduce_read(var, index or {}, reduce, progress, cancelled)
            if field is None or not np.isfinite(field).any():
                return None
            return float(np.nanmin(field)), float(np.nanmax(field))
        whole = sweep is None
        if whole:
            sweep = v.dims[0]
        n = self.file_size(sweep) if whole else self.dims[sweep].size
        lo, hi = np.inf, -np.inf
        for i in range(n):
            if cancelled is not None and cancelled():
                return None
            if whole:
                # straight from disk: a whole-variable scan would flush the slice cache
                block = self._read_uncached(v, {d: (i if d == sweep else (0, self.file_size(d))) for d in v.dims})
            elif reduce:
                part = None if progress is None else (lambda f, i=i: progress((i + f) / n))
                block = self.reduce_read(var, {**(index or {}), sweep: i}, reduce, part, cancelled)
                if block is None:
                    return None
            else:
                block = self.read(var, {**(index or {}), sweep: i})  # cached: an animation reads these next
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if np.isfinite(block).any():
                    lo = min(lo, float(np.nanmin(block)))
                    hi = max(hi, float(np.nanmax(block)))
            if progress is not None:
                progress((i + 1) / n)
        if not np.isfinite(lo):
            return None
        return lo, hi

    def close(self) -> None:
        for ds in self._datasets:
            ds.close()


def _partial(a: np.ndarray, axes: tuple[int, ...], stat: str, weights: np.ndarray | None = None):
    """One block's share of a statistic over ``axes``, ignoring NaN.

    The extreme for min and max; for sum and mean the total and the weight of the
    points that went into it (their number, if there are no ``weights``).
    """
    if stat in ("min", "max"):
        return (np.fmin if stat == "min" else np.fmax).reduce(a, axis=axes, initial=np.nan), None
    valid = ~np.isnan(a)
    values = np.where(valid, a, 0.0)
    if weights is None:
        return values.sum(axis=axes), valid.sum(axis=axes, dtype=np.float64)
    return (values * weights).sum(axis=axes), (valid * weights).sum(axis=axes)


def _merge(acc, part, stat: str):
    if stat in ("min", "max"):
        return (np.fmin if stat == "min" else np.fmax)(acc[0], part[0]), None
    return acc[0] + part[0], acc[1] + part[1]


def _finish(acc, stat: str) -> np.ndarray:
    """The statistic from its running totals: NaN where no value went in."""
    total, weight = acc
    if stat in ("min", "max"):
        return np.asarray(total)
    some = weight > 0
    return np.where(some, total / np.where(some, weight, 1.0) if stat == "mean" else total, np.nan)


def _attrs(obj) -> dict:
    return {k: obj.getncattr(k) for k in obj.ncattrs()}


def _values(raw) -> np.ndarray:
    """A float64 array with NaN where the data is masked (fill values, valid range)."""
    if isinstance(raw, np.ma.MaskedArray):
        return np.ma.filled(raw.astype(np.float64), np.nan)
    return np.asarray(raw, dtype=np.float64)


def _decode_times(values: np.ndarray, attrs: Mapping) -> list | None:
    units = attrs.get("units")
    if not units or " since " not in str(units):
        return None
    try:
        import cftime

        cal = attrs.get("calendar", "standard")
        finite = np.where(np.isfinite(values), values, 0)
        dates = cftime.num2date(finite, units, cal, only_use_cftime_datetimes=True)
        return list(np.atleast_1d(dates))
    except Exception:  # noqa: BLE001 - undecodable times fall back to raw numbers
        return None
