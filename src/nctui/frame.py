"""Build one frame: read the data a :class:`Request` describes, fit it to the screen, draw it.

This runs in a worker thread, so it only uses the immutable request and the
(thread-safe) source.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import coastlines, colormaps
from .data import TIME, Dim, Source, format_number, same_units
from .reduce import METHOD_LABELS, AxisMap, output_shape, resample
from .render import (
    ACCENT,
    BLACK,
    COAST,
    WHITE,
    AxisSpec,
    ColourBar,
    Line,
    LineFigure,
    _luminance,
    colour_indices,
    degree_ticks,
    image_area,
    line_figure,
    map_figure,
    text_block,
    y_label_width,
)
from .state import Colour, Request

MIN_WIDTH, MIN_HEIGHT = 30, 8


@dataclass
class Frame:
    request: Request
    lines: list[Line]
    width: int
    height: int
    header: str = ""
    stats: str = ""
    status: str = ""
    vrange: tuple[float, float] | None = None
    colour: Colour | None = None  # set when an automatic colour choice was made
    maps: dict[str, AxisMap] = field(default_factory=dict)
    area: tuple[int, int] = (0, 0)
    image_col: int = 0
    image_row: int = 0
    pad_top: int = 0
    line_fig: LineFigure | None = None
    probe_value: float | None = None
    error: str | None = None


class Cancelled(Exception):
    """The work was cancelled part way."""


# --------------------------------------------------------------------------- helpers


def request_data(src: Source, req: Request, progress: Callable[[float], None] | None = None,
                 cancelled: Callable[[], bool] | None = None) -> np.ndarray:
    """The data ``req`` shows, from every data point: its part of the variable, reduced as it asks.

    ``progress`` and ``cancelled`` are for a reduction that has to be read (see
    :meth:`Source.reduce_read`); raises :class:`Cancelled` if it is stopped.
    """
    data = src.reduce_read(req.var, req.index_dict(), req.reduce, progress, cancelled)
    if data is None:
        raise Cancelled
    return data


def short_name(dim: str) -> str:
    return {"projection_x_coordinate": "x", "projection_y_coordinate": "y"}.get(dim, dim)


def pixel_centres(dim: Dim, m: AxisMap) -> np.ndarray:
    """Coordinate value at the centre of every output pixel of ``m`` (screen order)."""
    q = np.arange(m.out, dtype=float)
    if m.reverse:
        q = m.out - 1 - q
    frac = m.start + (q + 0.5) * m.n / max(m.out, 1) - 0.5
    vals = dim.values
    if dim.size >= 2:
        # extend by one cell at each end so pixels beyond the outer centres extrapolate
        idx = np.concatenate([[-1.0], np.arange(dim.size, dtype=float), [float(dim.size)]])
        vals = np.concatenate([[2 * vals[0] - vals[1]], vals, [2 * vals[-1] - vals[-2]]])
        return np.interp(frac, idx, vals)
    return np.full(m.out, float(vals[0]) if dim.size else 0.0)


def axis_spec(dim: Dim, centres: np.ndarray, probe: int | None = None) -> AxisSpec:
    if dim.dates is not None:
        return AxisSpec(title=dim.name, centres=centres,
                        label=lambda v, d=dim: d.label(d.nearest(v), with_lead=False), probe=probe,
                        tick_values=dim.time_ticks)
    if not dim.has_coord:
        return AxisSpec(title=f"{dim.name} (index)", centres=centres, label=lambda v: str(int(round(v))),
                        probe=probe, integer=True)
    return AxisSpec(title=dim.axis_title(), centres=centres, probe=probe,
                    label=dim.tick_label if dim.is_lat and dim.period else None,
                    wrap=dim.wrap_value if dim.period else None,
                    tick_values=degree_ticks if dim.is_lat or dim.is_lon else None)


def _stats(data: np.ndarray) -> tuple[float, float, float, int]:
    finite = np.isfinite(data)
    nnan = int(data.size - finite.sum())
    if not finite.any():
        return np.nan, np.nan, np.nan, nnan
    vals = data[finite]
    return float(vals.min()), float(vals.max()), float(vals.mean()), nnan


def decide_colour(c: Colour, lo: float, hi: float) -> Colour | None:
    """The automatic first choice of colour map for a variable."""
    if c.decided or not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
        return None  # wait for a slice that isn't constant (e.g. w at the surface)
    new = copy.copy(c)
    new.decided = True
    if np.isfinite(lo) and np.isfinite(hi) and lo < 0 < hi:
        if min(-lo, hi) >= 0.1 * max(-lo, hi):
            new.cmap = colormaps.DEFAULT_DIVERGING
            new.symmetric = True
    return new


def colour_range(c: Colour, data: np.ndarray, anim_range) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if c.mode in ("locked", "manual") and c.vmin is not None and c.vmax is not None:
        lo, hi = c.vmin, c.vmax
    else:
        if c.mode == "anim" and anim_range is not None:
            lo, hi = anim_range
        elif finite.size == 0:
            lo, hi = 0.0, 1.0
        elif c.clip:
            lo, hi = (float(v) for v in np.percentile(finite, [2, 98]))
        else:
            lo, hi = float(finite.min()), float(finite.max())
        if c.symmetric:
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
    if c.log:
        if hi <= 0:
            hi = 1.0
        if lo <= 0:
            pos = finite[finite > 0]
            lo = float(pos.min()) if pos.size else hi * 1e-6
    if hi == lo:
        pad = abs(lo) * 0.01 or 0.5
        lo, hi = lo - pad, hi + pad
    return float(lo), float(hi)


def range_label(c: Colour, scanning: bool = False, dim: str | None = None) -> str:
    if c.mode == "locked":
        text = "locked range"
    elif c.mode == "manual":
        text = "manual range"
    elif c.mode == "anim" and dim is not None:
        text = f"range of all {short_name(dim)} steps" + (" (scanning…)" if scanning else "")
    else:
        text = "auto range" + (" 2–98%" if c.clip else "")
    if c.symmetric and c.mode in ("auto", "anim"):
        text += " ±"
    if c.log:
        text += " · log"
    return text


def _reduction_text(name: str, m: AxisMap, total: int) -> str:
    of = f" of {total}" if m.n != total else ""
    if m.out >= m.n:
        k = m.upscale
        return f"{short_name(name)} {m.n}{of} " + ("1:1" if k == 1 else f"×{k}")
    return f"{short_name(name)} {m.n}{of}→{m.out} ÷{m.factor:.1f}"


def _value_text(v: float | None, units: str) -> str:
    if v is None or not np.isfinite(v):
        return "missing"
    return f"{format_number(v, 6)} {units}".rstrip()


# --------------------------------------------------------------------------- builders


def build_frame(src: Source, req: Request, width: int, height: int,
                progress: Callable[[float], None] | None = None,
                cancelled: Callable[[], bool] | None = None) -> Frame | None:
    """Draw ``req``. A reduction that isn't cached is read first, reporting ``progress``
    (0 to 1); returns None if ``cancelled`` stops it."""
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return Frame(req, text_block("make the window bigger", width, height), width, height)
    try:
        data = request_data(src, req, progress, cancelled)
        if req.kind == "map":
            return _map_frame(src, req, data, width, height)
        if req.kind == "line":
            return _line_frame(src, req, data, width, height)
        return _scalar_frame(src, req, data, width, height)
    except Cancelled:
        return None
    except Exception as exc:  # noqa: BLE001 - show any read/draw problem in the pane
        msg = f"can't draw {req.var}: {exc}"
        return Frame(req, text_block(msg, width, height), width, height, error=msg)


def _header(src: Source, req: Request, others: list[str], view: str) -> str:
    """The view, then each of ``others``: its position, or the statistic it is reduced with."""
    idx = req.index_dict()
    reduced = req.reduce_dict()
    parts = [view]
    for d in others:
        where = reduced[d] if d in reduced else src.dims[d].label(int(idx[d]))
        parts.append(f"{short_name(d)} {where}")
    return " · ".join(parts)


def _weighting(src: Source, req: Request) -> str:
    """For the status line: how a mean over dimensions was weighted."""
    note = src.reduce_weighting(req.var, req.reduce)
    return f" · {note}" if note else ""


def _map_frame(src: Source, req: Request, data: np.ndarray, width: int, height: int) -> Frame:
    v = src.vars[req.var]
    idx = req.index_dict()
    xd, yd = src.dims[req.xdim], src.dims[req.ydim]
    kept = [d for d in v.dims if isinstance(idx.get(d), tuple)]
    if kept.index(req.xdim) < kept.index(req.ydim):
        data = data.T  # → [y, x]
    xa, xb = idx[req.xdim]
    ya, yb = idx[req.ydim]
    nx, ny = xb - xa, yb - ya
    lo, hi, mean, nnan = _stats(data)

    colour = req.colour
    decided = decide_colour(colour, lo, hi)
    if decided is not None:
        colour = decided
    vmin, vmax = colour_range(colour, data, req.anim_range)

    # layout
    yspec0 = axis_spec(yd, yd.values[ya:yb])
    ylw = y_label_width(yspec0)
    px, py = image_area(width, height, ylw)
    if px < 4 or py < 4:
        return Frame(req, text_block("make the window bigger", width, height), width, height)
    if same_units(xd, yd):
        sx = (abs(xd.values[xb - 1] - xd.values[xa]) * nx / (nx - 1)) if nx > 1 else 1.0
        sy = (abs(yd.values[yb - 1] - yd.values[ya]) * ny / (ny - 1)) if ny > 1 else 1.0
        aspect = sx / sy if sy > 0 and sx > 0 else 1.0
    else:
        aspect = nx / ny  # no common scale (e.g. metres against level numbers): square grid cells
    out_w, out_h = output_shape(nx, ny, px, py, req.equal, aspect)
    opts = req.options
    x_rev = (not xd.increasing()) != opts.flip_x
    y_rev = (yd.increasing() != (yd.positive == "down")) != opts.flip_y
    xmap = AxisMap(xa, nx, out_w, reverse=x_rev)
    ymap = AxisMap(ya, ny, out_h, reverse=y_rev)

    img = resample(data, xmap, ymap, opts.method)
    pix = colour_indices(img, vmin, vmax, colour.log)

    xc, yc = pixel_centres(xd, xmap), pixel_centres(yd, ymap)
    if opts.coastlines and xd.is_lon and yd.is_lat:
        pix[coastlines.rasterise(xc, yc, lat_wraps=yd.period is not None)] = COAST

    probe = req.probe_dict()
    pxp, pyp = probe.get(req.xdim), probe.get(req.ydim)
    probe_x = probe_y = None
    if pxp is not None and pyp is not None:
        sx_ = xmap.pixel_span(pxp)
        sy_ = ymap.pixel_span(pyp)
        if sx_ and sy_:
            probe_x = (sx_[0] + sx_[1]) // 2
            probe_y = (sy_[0] + sy_[1]) // 2
            # a single pixel, or a small cross when each data point is a big block
            big = xmap.upscale >= 4 and ymap.upscale >= 4
            marks = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)) if big else ((0, 0),)
            lum = _luminance(colour.cmap)
            for ddx, ddy in marks:
                yy, xx = probe_y + ddy, probe_x + ddx
                if 0 <= yy < out_h and 0 <= xx < out_w:
                    under = int(pix[yy, xx])
                    pix[yy, xx] = BLACK if (lum[under] if under >= 0 else 0.0) > 0.5 else WHITE

    if req.selection is not None:
        x0, y0, x1, y1 = req.selection
        x0, x1 = sorted((int(np.clip(x0, 0, out_w - 1)), int(np.clip(x1, 0, out_w - 1))))
        y0, y1 = sorted((int(np.clip(y0, 0, out_h - 1)), int(np.clip(y1, 0, out_h - 1))))
        pix[y0, x0:x1 + 1] = ACCENT
        pix[y1, x0:x1 + 1] = ACCENT
        pix[y0:y1 + 1, x0] = ACCENT
        pix[y0:y1 + 1, x1] = ACCENT

    xspec = axis_spec(xd, xc, probe_x)
    yspec = axis_spec(yd, yc, probe_y)
    bar = ColourBar(colour.cmap, vmin, vmax, colour.log, units=v.units)
    fig = map_figure(pix, colour.cmap, xspec, yspec, bar, width, height)

    free = [d for d in v.dims if d not in (req.xdim, req.ydim)]
    header = _header(src, req, free, _view_name(src, req))
    stats = f"min {format_number(lo)}  max {format_number(hi)}  mean {format_number(mean)}"
    if nnan:
        stats += f"  missing {nnan}"
    stats += f" · {range_label(colour, req.anim_range is None, req.anim_dim)} · {colour.cmap}"

    reduced = xmap.out < xmap.n or ymap.out < ymap.n
    status = f"{_reduction_text(req.xdim, xmap, xd.size)} · {_reduction_text(req.ydim, ymap, yd.size)}"
    if reduced:
        status += f" ({METHOD_LABELS[opts.method]})"
    status += _weighting(src, req)
    probe_value = None
    if pxp is not None and pyp is not None and xa <= pxp < xb and ya <= pyp < yb:
        probe_value = float(data[pyp - ya, pxp - xa])
        status += (f" · probe {short_name(req.xdim)}={xd.label(pxp)} {short_name(req.ydim)}={yd.label(pyp)}"
                   f" → {_value_text(probe_value, v.units)}")
    return Frame(
        request=req,
        lines=fig.lines,
        width=width,
        height=height,
        header=header,
        stats=stats,
        status=status,
        vrange=(vmin, vmax),
        colour=decided,
        maps={req.xdim: xmap, req.ydim: ymap},
        area=(px, py),
        image_col=fig.image_col,
        image_row=fig.image_row,
        pad_top=fig.pad_top,
        probe_value=probe_value,
    )


def _view_name(src: Source, req: Request) -> str:
    if req.kind == "map":
        rx, ry = src.dims[req.xdim].role, src.dims[req.ydim].role
        roles = {rx, ry}
        name = "map" if roles == {"x", "y"} else ("section" if "z" in roles else "plot")
        return f"{name} {short_name(req.ydim)}–{short_name(req.xdim)}"
    if req.kind == "line":
        role = src.dims[req.xdim].role
        kind = "profile" if role == "z" else ("time series" if role == TIME else "line")
        return f"{kind} along {short_name(req.xdim)}"
    return "value"


def _line_frame(src: Source, req: Request, data: np.ndarray, width: int, height: int) -> Frame:
    v = src.vars[req.var]
    idx = req.index_dict()
    d = src.dims[req.xdim]
    a, b = idx[req.xdim]
    values = np.asarray(data, dtype=float).reshape(-1)
    coord = d.values[a:b]
    lo, hi, mean, nnan = _stats(values)
    colour = req.colour
    value_range = None
    if colour.mode in ("locked", "manual") and colour.vmin is not None:
        value_range = (colour.vmin, colour.vmax)
    elif colour.mode == "anim" and req.anim_range is not None:
        value_range = req.anim_range
    vertical = d.role == "z"
    coord_up = (d.positive != "down") != req.options.flip_y
    marker_i = req.probe_dict().get(req.xdim)
    marker = marker_i - a if marker_i is not None and a <= marker_i < b else None
    title = f"{v.name} ({v.units})" if v.units else v.name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fig = line_figure(coord, values, width, height, axis_spec(d, coord), title,
                          vertical=vertical, coord_up=coord_up, marker=marker, value_range=value_range)
    here = [x for x in v.dims if x in idx and x != req.xdim and src.dims[x].role in ("x", "y")]
    where = ", ".join(f"{short_name(x)}={src.dims[x].label(int(idx[x]), with_lead=False)}" for x in here)
    view = _view_name(src, req) + (f" at {where}" if where else "")
    others = [x for x in v.dims if x != req.xdim and x not in here]
    header = _header(src, req, others, view)
    stats = f"min {format_number(lo)}  max {format_number(hi)}  mean {format_number(mean)}"
    if nnan:
        stats += f"  missing {nnan}"
    if value_range is not None:
        stats += f" · {range_label(colour, False, req.anim_dim)}"
    n = b - a
    of = f" of {d.size}" if n != d.size else ""
    dots = fig.coord_dots
    status = f"{short_name(req.xdim)} {n}{of} points on {dots} dots" + (" 1:1" if dots >= n else f" ÷{n / dots:.1f}")
    status += _weighting(src, req)
    probe_value = None
    if marker is not None:
        probe_value = float(values[marker])
        status += f" · {short_name(req.xdim)}={d.label(marker_i)} → {_value_text(probe_value, v.units)}"
    maps = {req.xdim: AxisMap(a, n, max(1, dots), reverse=False)}
    return Frame(
        request=req,
        lines=fig.lines,
        width=width,
        height=height,
        header=header,
        stats=stats,
        status=status,
        vrange=(lo, hi) if np.isfinite(lo) else None,
        maps=maps,
        area=(2 * fig.cols, 4 * fig.rows),
        image_col=fig.plot_col,
        image_row=fig.plot_row,
        line_fig=fig,
        probe_value=probe_value,
    )


def _scalar_frame(src: Source, req: Request, data: np.ndarray, width: int, height: int) -> Frame:
    v = src.vars[req.var]
    value = float(np.asarray(data).reshape(-1)[0])
    text = f"{v.name} = {_value_text(value, v.units)}"
    return Frame(req, text_block(text, width, height), width, height, header=_header(src, req, list(v.dims), "value"),
                 stats=v.long_name, status=_weighting(src, req).removeprefix(" · "), probe_value=value)
