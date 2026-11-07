"""What is being looked at, and every way the user can change it.

:class:`Viewer` holds the current variable, which dimensions are plotted, the
position along every other dimension, zoom windows, the probe, and colour and
plot settings. It knows nothing about the terminal: the app calls its methods
in response to keys and asks it for a :class:`Request` describing what to draw.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np

from . import colormaps
from .data import OTHER, TIME, Index, Source, Var, X, Y, Z, same_units
from .reduce import AxisMap, centred_window, every_point_lengths

# (x role, y role) of the preset views, in the order `v` cycles through them
PRESET_VIEWS: list[tuple[str, str]] = [(X, Y), (X, Z), (Y, Z)]
RANGE_MODES = ("auto", "locked", "anim", "manual")
ASPECTS = ("auto", "equal", "fill")
_STEP_PRIORITY = {TIME: 0, Z: 1, OTHER: 2, Y: 3, X: 4}


@dataclass
class Colour:
    """Colour settings for one variable."""

    cmap: str = colormaps.DEFAULT
    mode: str = "auto"  # auto | locked | anim (one range for every animation step) | manual
    vmin: float | None = None
    vmax: float | None = None
    symmetric: bool = False
    log: bool = False
    clip: bool = False  # clip to 2nd–98th percentile (auto mode)
    decided: bool = False  # the automatic choice of map has been made (or the user chose)


@dataclass
class Options:
    """Plot settings shared by all variables."""

    aspect: str = "auto"  # auto | equal | fill
    flip_x: bool = False
    flip_y: bool = False
    method: str = "mean"
    coastlines: bool = False


@dataclass(frozen=True)
class Request:
    """An immutable description of one frame to draw."""

    var: str
    kind: str  # "map" | "line" | "scalar"
    xdim: str | None  # map: x axis; line: the line's dimension
    ydim: str | None  # map: y axis
    index: tuple[tuple[str, Index], ...]
    probe: tuple[tuple[str, int], ...]
    colour: Colour
    options: Options
    equal: bool
    anim_range: tuple[float, float] | None = None  # "anim" colour mode: range over every step
    anim_dim: str | None = None  # the dimension an animation runs along
    selection: tuple[int, int, int, int] | None = None  # pixel box while dragging

    def index_dict(self) -> dict[str, Index]:
        return dict(self.index)

    def probe_dict(self) -> dict[str, int]:
        return dict(self.probe)


class Viewer:
    def __init__(self, source: Source):
        self.src = source
        self.var: str | None = None
        self.kind = "none"
        self.xdim: str | None = None
        self.ydim: str | None = None
        # positions and zoom windows are kept per role (time, z, y, x) so they carry
        # over between variables on different dimensions (full_levels ↔ half_levels)
        self._pos: dict[str, tuple[str, int]] = {}
        self._win: dict[str, tuple[str, int, int]] = {}
        self.stack: list[tuple[str, str | None, str | None]] = []
        self.colours: dict[str, Colour] = {}
        self.options = Options()
        # "anim" colour ranges, keyed by anim_range_key()
        self.anim_ranges: dict[tuple, tuple[float, float]] = {}
        self.anim_dim: str | None = None  # the dimension last animated along
        self.selection: tuple[int, int, int, int] | None = None
        # filled in by the app from the last drawn frame
        self.maps: dict[str, AxisMap] = {}
        self.area: tuple[int, int] = (0, 0)
        self.last_range: tuple[float, float] | None = None

    # ------------------------------------------------------------------ basics

    @property
    def v(self) -> Var:
        assert self.var is not None
        return self.src.vars[self.var]

    def role(self, dim: str) -> str:
        return self.src.dims[dim].role

    def _key(self, dim: str) -> str:
        r = self.role(dim)
        return dim if r == OTHER else r

    def role_dim(self, role: str) -> str | None:
        if self.var is None:
            return None
        return next((d for d in self.v.dims if self.role(d) == role), None)

    def plotted(self) -> list[str]:
        if self.kind == "map":
            return [d for d in (self.xdim, self.ydim) if d]
        if self.kind == "line":
            return [self.xdim] if self.xdim else []
        return []

    def free_dims(self) -> list[str]:
        if self.var is None:
            return []
        p = self.plotted()
        return [d for d in self.v.dims if d not in p]

    # ------------------------------------------------------------------ positions

    def pos(self, dim: str) -> int:
        """Current index along ``dim`` (carried over by coordinate value from a sibling dim)."""
        d = self.src.dims[dim]
        entry = self._pos.get(self._key(dim))
        if entry is None:
            return d.size // 2 if self.role(dim) in (X, Y) else 0
        name, i = entry
        if name == dim:
            return int(np.clip(i, 0, d.size - 1))
        other = self.src.dims.get(name)
        if other is None or not (other.has_coord and d.has_coord):
            return int(np.clip(i, 0, d.size - 1))
        return d.nearest(other.values[int(np.clip(i, 0, other.size - 1))])

    def set_pos(self, dim: str, i: int) -> bool:
        size = self.src.dims[dim].size
        i = int(np.clip(i, 0, size - 1))
        changed = self.pos(dim) != i
        self._pos[self._key(dim)] = (dim, i)
        if dim in self.plotted():
            self._keep_in_window(dim, i)
        return changed

    def step(self, dim: str | None, delta: int) -> bool:
        if dim is None:
            return False
        return self.set_pos(dim, self.pos(dim) + delta)

    def step_targets(self) -> tuple[str | None, str | None]:
        """The dimensions that ``[ ]`` and ``{ }`` step."""
        free = self.free_dims()
        if not free:
            return None, None
        order = sorted(free, key=lambda d: (_STEP_PRIORITY[self.role(d)], self.v.dims.index(d)))
        a = next((d for d in order if self.role(d) == TIME), order[0])
        rest = [d for d in order if d != a]
        b = next((d for d in rest if self.role(d) == Z), rest[0] if rest else None)
        return a, b

    # ------------------------------------------------------------------ windows

    def window(self, dim: str) -> tuple[int, int]:
        d = self.src.dims[dim]
        entry = self._win.get(self._key(dim))
        if entry is None:
            return 0, d.size
        name, a, b = entry
        if name == dim:
            a, b = int(np.clip(a, 0, d.size - 1)), int(np.clip(b, 1, d.size))
            return (a, b) if b > a else (0, d.size)
        other = self.src.dims.get(name)
        if other is None or not (other.has_coord and d.has_coord):
            return 0, d.size
        i0 = d.nearest(other.values[a])
        i1 = d.nearest(other.values[b - 1])
        lo, hi = min(i0, i1), max(i0, i1) + 1
        return lo, hi

    def set_window(self, dim: str, a: int, b: int) -> None:
        size = self.src.dims[dim].size
        a, b = int(np.clip(a, 0, size - 1)), int(np.clip(b, 1, size))
        if b <= a:
            b = min(size, a + 1)
        if a == 0 and b == size:
            self._win.pop(self._key(dim), None)
        else:
            self._win[self._key(dim)] = (dim, a, b)

    def _keep_in_window(self, dim: str, i: int) -> None:
        a, b = self.window(dim)
        if a <= i < b:
            return
        length = b - a
        if i < a:
            self.set_window(dim, i, i + length)
        else:
            self.set_window(dim, i - length + 1, i + 1)

    def is_zoomed(self) -> bool:
        return any(self.window(d) != (0, self.src.dims[d].size) for d in self.plotted())

    # ------------------------------------------------------------------ variables & views

    def select_var(self, name: str) -> None:
        v = self.src.vars[name]
        prev_kind, prev_x, prev_y = self.kind, self.xdim, self.ydim
        prev_roles = (self.role(prev_x) if prev_x else None, self.role(prev_y) if prev_y else None)
        self.var = name
        if v.ndim == 0:
            self.kind, self.xdim, self.ydim = "scalar", None, None
            self.stack = []
            return
        if v.ndim == 1:
            self.kind, self.xdim, self.ydim = "line", v.dims[0], None
            self.stack = []
            return
        if prev_kind == "line" and self.stack and prev_roles[0] is not None:
            line_dim = prev_x if prev_x in v.dims else self.role_dim(prev_roles[0])
            if line_dim is not None and prev_roles[0] != OTHER:
                self.kind, self.xdim, self.ydim = "line", line_dim, None
                return
        self.stack = []
        self.kind = "map"
        self.xdim, self.ydim = self._resolve_view(prev_x, prev_y, prev_roles)

    def _resolve_view(self, prev_x, prev_y, prev_roles) -> tuple[str, str]:
        v = self.v
        if prev_x in v.dims and prev_y in v.dims and prev_x != prev_y:
            return prev_x, prev_y
        if all(r is not None and r != OTHER for r in prev_roles):
            x, y = self.role_dim(prev_roles[0]), self.role_dim(prev_roles[1])
            if x and y:
                return x, y
        views = self.available_views()
        return views[0]

    def available_views(self) -> list[tuple[str, str]]:
        if self.var is None or self.v.ndim < 2:
            return []
        out = []
        for xr_, yr_ in PRESET_VIEWS:
            x, y = self.role_dim(xr_), self.role_dim(yr_)
            if x and y:
                out.append((x, y))
        if not out:
            out.append((self.v.dims[-1], self.v.dims[-2]))
        return out

    def view_name(self) -> str:
        if self.kind == "map":
            rx, ry = self.role(self.xdim), self.role(self.ydim)
            name = "map" if {rx, ry} == {X, Y} else ("section" if Z in (rx, ry) else "plot")
            return f"{name} {self.ydim}–{self.xdim}"
        if self.kind == "line":
            return ("profile" if self.role(self.xdim) == Z else
                    "time series" if self.role(self.xdim) == TIME else "line") + f" along {self.xdim}"
        return "value"

    def cycle_view(self) -> bool:
        if self.kind == "line" and self.stack:
            self.back()
        if self.kind != "map":
            return False
        views = self.available_views()
        if len(views) < 2:
            return False
        cur = (self.xdim, self.ydim)
        i = views.index(cur) if cur in views else -1
        self.xdim, self.ydim = views[(i + 1) % len(views)]
        return True

    def swap_axes(self) -> bool:
        if self.kind != "map":
            return False
        self.xdim, self.ydim = self.ydim, self.xdim
        return True

    def set_axes(self, x: str, y: str | None) -> None:
        """Put ``x`` (and ``y``) on the plot; ``y=None`` makes a line plot along ``x``."""
        v = self.v
        if x not in v.dims or (y is not None and (y not in v.dims or y == x)):
            raise ValueError("choose two different dimensions of this variable")
        self.stack = []
        if y is None:
            self.kind, self.xdim, self.ydim = "line", x, None
        else:
            self.kind, self.xdim, self.ydim = "map", x, y

    def assign_axis(self, dim: str, axis: str) -> bool:
        """Put ``dim`` on the ``x`` or ``y`` axis of the current map, swapping if needed."""
        if self.kind != "map" or dim not in self.v.dims:
            return False
        x, y = self.xdim, self.ydim
        if axis == "x":
            x, y = (dim, x if dim == y else y)
        else:
            x, y = (y if dim == x else x, dim)
        self.xdim, self.ydim = x, y
        return True

    def open_line(self, dim: str) -> bool:
        if self.kind != "map" or dim not in self.v.dims:
            return False
        self.stack.append((self.kind, self.xdim, self.ydim))
        self.kind, self.xdim, self.ydim = "line", dim, None
        return True

    def back(self) -> bool:
        if not self.stack:
            return False
        kind, x, y = self.stack.pop()
        v = self.v
        roles = (self.role(x) if x in self.src.dims else None, self.role(y) if y in self.src.dims else None)
        self.kind = kind
        if v.ndim >= 2:
            self.xdim, self.ydim = self._resolve_view(x, y, roles)
        return True

    # ------------------------------------------------------------------ probe & zoom

    def probe_step(self, dim: str) -> int:
        m = self.maps.get(dim)
        return max(1, int(round(m.factor))) if m else 1

    def move_probe(self, dx: int, dy: int) -> bool:
        """Move the probe by screen pixels: +dx is right, +dy is up."""
        changed = False
        if self.kind == "map":
            for dim, delta, axis in ((self.xdim, dx, "x"), (self.ydim, dy, "y")):
                if not delta or dim is None:
                    continue
                m = self.maps.get(dim)
                sign = 1
                if m is not None:
                    if axis == "x":
                        sign = -1 if m.reverse else 1
                    else:
                        sign = 1 if m.reverse else -1
                changed |= self.step(dim, delta * sign * self.probe_step(dim))
        elif self.kind == "line" and self.xdim:
            d = self.src.dims[self.xdim]
            if self.line_is_vertical():
                delta = dy or dx  # up (or right) moves up the profile
                sign = 1 if d.increasing() == self.coord_up() else -1
            else:
                delta = dx or dy  # right (or up) moves along the line
                sign = 1 if d.increasing() else -1
            changed |= self.step(self.xdim, delta * sign * self.probe_step(self.xdim))
        return changed

    def line_is_vertical(self) -> bool:
        return self.kind == "line" and self.xdim is not None and self.role(self.xdim) == Z

    def coord_up(self) -> bool:
        """For a profile: does the vertical coordinate increase upwards on screen?"""
        d = self.src.dims[self.xdim]
        return (d.positive != "down") != self.options.flip_y

    def set_probe_pixel(self, px: int | None, py: int | None) -> bool:
        changed = False
        if self.kind == "map":
            if px is not None and self.xdim in self.maps:
                changed |= self.set_pos(self.xdim, self.maps[self.xdim].pixel_to_index(px))
            if py is not None and self.ydim in self.maps:
                changed |= self.set_pos(self.ydim, self.maps[self.ydim].pixel_to_index(py))
        return changed

    def zoom(self, factor: float) -> bool:
        changed = False
        for dim in self.plotted():
            n = self.src.dims[dim].size
            a, b = self.window(dim)
            length = int(np.clip(round((b - a) / factor), min(n, 3), n))
            if length == b - a and factor > 1:
                continue
            na, nb = centred_window(n, length, self.pos(dim))
            if (na, nb) != (a, b):
                self.set_window(dim, na, nb)
                changed = True
        return changed

    def fit(self) -> bool:
        changed = self.is_zoomed()
        for dim in self.plotted():
            self._win.pop(self._key(dim), None)
        return changed

    def point_aspect(self) -> float:
        """Width/height of one grid cell in data units (for equal-aspect maps).

        When the axes have different units there is no common scale, so equal aspect
        draws every grid cell square instead (aspect 1).
        """
        if self.kind != "map" or not same_units(self.src.dims[self.xdim], self.src.dims[self.ydim]):
            return 1.0
        sx = _spacing(self.src.dims[self.xdim].values)
        sy = _spacing(self.src.dims[self.ydim].values)
        if sx and sy:
            return sx / sy
        return 1.0

    def every_point(self) -> bool:
        """Zoom so every data point gets at least one pixel, centred on the probe."""
        px, py = self.area
        if px <= 0 or py <= 0:
            return False
        if self.kind == "map":
            nx, ny = self.src.dims[self.xdim].size, self.src.dims[self.ydim].size
            wx, wy = every_point_lengths(nx, ny, px, py, self.equal_aspect(), self.point_aspect())
            for dim, w in ((self.xdim, wx), (self.ydim, wy)):
                n = self.src.dims[dim].size
                self.set_window(dim, *centred_window(n, w, self.pos(dim)))
            return True
        if self.kind == "line" and self.xdim:
            n = self.src.dims[self.xdim].size
            w = min(n, px if self.role(self.xdim) != Z else py)
            self.set_window(self.xdim, *centred_window(n, w, self.pos(self.xdim)))
            return True
        return False

    def zoom_to_pixels(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        if self.kind != "map" or self.xdim not in self.maps or self.ydim not in self.maps:
            return False
        if abs(x1 - x0) < 1 and abs(y1 - y0) < 1:
            return False
        for dim, p0, p1 in ((self.xdim, x0, x1), (self.ydim, y0, y1)):
            m = self.maps[dim]
            i0, i1 = m.pixel_to_index(p0), m.pixel_to_index(p1)
            lo, hi = min(i0, i1), max(i0, i1) + 1
            if hi - lo >= 2:
                self.set_window(dim, lo, hi)
                self.set_pos(dim, (lo + hi - 1) // 2)
        return True

    # ------------------------------------------------------------------ centring (periodic axes)

    def can_centre(self, dim: str | None) -> bool:
        return dim is not None and self.src.period(dim) is not None

    def centre_on(self, dim: str, value: float | None) -> None:
        """Wrap ``dim`` round so ``value`` is in the middle of the plot (None: as in the file).

        The probe stays on the same grid point and a zoom keeps its size, centred on it.
        """
        src = self.src
        n = src.dims[dim].size
        stored = (self.pos(dim) + src.roll(dim)) % n
        a, b = self.window(dim)
        src.set_centre(dim, value)
        i = (stored - src.roll(dim)) % n
        self._pos[self._key(dim)] = (dim, i)
        if b - a < n:
            self.set_window(dim, *centred_window(n, b - a, i))
        else:
            self._win.pop(self._key(dim), None)
        self.anim_ranges.clear()  # windows now mean different points

    def centring_axes(self, latitude: bool) -> list[str]:
        """The plotted axes "centre here" wraps round; latitude only if asked."""
        if self.kind != "map":
            return []
        return [d for d in (self.xdim, self.ydim)
                if self.can_centre(d) and (latitude or not self.src.dims[d].is_lat)]

    def centre_here(self, latitude: bool = False) -> list[str]:
        """Centre the plotted axes that can wrap round on the probe. Returns those axes."""
        axes = self.centring_axes(latitude)
        for d in axes:
            dim = self.src.dims[d]
            self.centre_on(d, dim.wrap_value(float(dim.values[self.pos(d)])))
        return axes

    def centred_axes(self) -> list[str]:
        return [d for d in self.plotted() if self.src.centre(d) is not None]

    def uncentre(self) -> bool:
        axes = self.centred_axes()
        for d in axes:
            self.centre_on(d, None)
        return bool(axes)

    # ------------------------------------------------------------------ colours

    @property
    def colour(self) -> Colour:
        assert self.var is not None
        return self.colours.setdefault(self.var, Colour())

    def cycle_cmap(self, step: int = 1) -> str:
        c = self.colour
        c.cmap = colormaps.next_name(c.cmap, step)
        c.decided = True
        return c.cmap

    def toggle_lock(self) -> str:
        c = self.colour
        if c.mode == "auto":
            if self.last_range is None:
                return c.mode
            c.mode = "locked"
            c.vmin, c.vmax = self.last_range
        else:
            c.mode = "auto"
        return c.mode

    def set_manual_range(self, vmin: float, vmax: float) -> None:
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin >= vmax:
            raise ValueError("minimum must be less than maximum")
        c = self.colour
        c.mode, c.vmin, c.vmax = "manual", float(vmin), float(vmax)

    def equal_aspect(self) -> bool:
        if self.kind != "map":
            return False
        if self.options.aspect != "auto":
            return self.options.aspect == "equal"
        return same_units(self.src.dims[self.xdim], self.src.dims[self.ydim])

    def can_show_coastlines(self) -> bool:
        if self.kind != "map":
            return False
        dx, dy = self.src.dims[self.xdim], self.src.dims[self.ydim]
        return dx.is_lon and dy.is_lat

    # ------------------------------------------------------------------ request

    def sweep_dim(self) -> str | None:
        """The dimension an animation runs along: the last one animated, else what space would pick."""
        if self.anim_dim in self.free_dims():
            return self.anim_dim
        return self.step_targets()[0]

    def _index(self) -> tuple[tuple[str, Index], ...]:
        plotted = self.plotted()
        return tuple((d, self.window(d) if d in plotted else self.pos(d)) for d in self.v.dims)

    def anim_range_key(self) -> tuple | None:
        """What the "anim" colour range depends on: the view, and the position of every other dim."""
        dim = self.sweep_dim()
        if self.var is None or dim is None:
            return None
        return (self.var, dim, tuple(item for item in self._index() if item[0] != dim))

    def request(self) -> Request | None:
        if self.var is None:
            return None
        plotted = self.plotted()
        index = self._index()
        probe = tuple((d, self.pos(d)) for d in plotted)
        key = self.anim_range_key()
        return Request(
            var=self.var,
            kind=self.kind,
            xdim=self.xdim,
            ydim=self.ydim,
            index=index,
            probe=probe,
            colour=copy.copy(self.colour),
            options=replace(self.options),
            equal=self.equal_aspect(),
            anim_range=self.anim_ranges.get(key) if key is not None else None,
            anim_dim=self.sweep_dim(),
            selection=self.selection,
        )

    def position_label(self, dim: str) -> str:
        return self.src.dims[dim].label(self.pos(dim))


def _spacing(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    d = np.abs(np.diff(values))
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if len(d) else None
