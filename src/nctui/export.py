"""Save the current view as an image with matplotlib, using every data point.

A reduced view (a time mean, a zonal mean…) is reduced again from the file's data,
or taken from the source's cache of it: never from the pixels on screen.

matplotlib is imported only here, so starting nctui doesn't pay for it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from . import coastlines
from .data import Dim, Source, same_units
from .frame import Cancelled, _view_name, request_data, short_name
from .state import Request


def _clean(text: str) -> str:
    text = text.replace(" ", "T", 1) if re.match(r"\d{4}-\d\d-\d\d \d", text) else text
    return re.sub(r"[^A-Za-z0-9._+-]+", "", text.replace(":", ""))


def default_filename(src: Source, req: Request, skip: str | None = None) -> str:
    """``./var_position_position.png`` for the current view (leaving out dimension ``skip``)."""
    v = src.vars[req.var]
    idx = req.index_dict()
    reduced = req.reduce_dict()
    parts = [req.var]
    if req.kind == "line":
        parts.append(_view_name(src, req).split(" ")[0].replace("time", "series"))
    for d in v.dims:
        ix = idx.get(d)
        if d in reduced:
            parts.append(f"{short_name(d)}{reduced[d]}")
        elif isinstance(ix, int) and d != skip:
            dim = src.dims[d]
            label = dim.label(ix, with_lead=False)
            parts.append(_clean(label) if dim.dates is not None else f"{short_name(d)}{_clean(label)}")
    name = "_".join(p for p in parts if p)[:120]
    return f"./{name}.png"


def default_title(src: Source, req: Request, header: str) -> str:
    """The variable, how a mean over dimensions was weighted if it was, then ``header``
    (the view, with each position or reduction)."""
    v = src.vars[req.var]
    first = v.long_name or v.name
    if v.units:
        first += f" ({v.units})"
    weighting = src.reduce_weighting(req.var, req.reduce)
    if weighting.startswith("cos"):
        first += f", {weighting}"
    return f"{first}\n{header}" if header else first


def _edges(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[:-1] + c[1:]) / 2
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def _time_ticks(axis, dim: Dim, values: np.ndarray, max_ticks: int = 6) -> None:
    """Tick a matplotlib axis at the data's own time steps, as the TUI does."""
    ticks = dim.time_ticks(float(np.nanmin(values)), float(np.nanmax(values)), max_ticks)
    if ticks:
        axis.set_ticks(ticks, labels=[dim.label(dim.nearest(t), with_lead=False) for t in ticks])


def _wrapped_labels(axis, dim: Dim) -> None:
    """A centred axis runs on past its period (135…495°); label it as it reads (…180, -180…)."""
    from matplotlib.ticker import FuncFormatter

    axis.set_major_formatter(FuncFormatter(lambda v, pos: dim.tick_label(v)))


def _degree_locator(axis) -> None:
    """Latitude/longitude ticks at steps that land on 0, ±90 and ±180 (15°, 30°, 45°, 90°…)."""
    from matplotlib.ticker import MaxNLocator

    axis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 1.5, 3, 4.5, 9, 10]))


def _cell_aspect(xd: Dim, yd: Dim) -> float | str:
    """matplotlib aspect for an equal-aspect plot: true scale, or square cells if units differ."""
    if same_units(xd, yd):
        return "equal"
    sx, sy = np.median(np.abs(np.diff(xd.values))), np.median(np.abs(np.diff(yd.values)))
    return float(sx / sy) if xd.size > 1 and yd.size > 1 and sx > 0 and sy > 0 else "auto"


def _check_folder(path: str) -> str:
    path = os.path.expanduser(path)
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        raise OSError(f"folder {folder} doesn't exist")
    return path


def _new_figure(size: tuple[float, float], dpi: int):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, dpi=dpi, layout="constrained")
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot()


def _fit_title(text: str, width_in: float, fontsize: float = 10) -> str:
    """Wrap a title's lines to the figure's width (matplotlib clips long ones)."""
    import textwrap

    chars = max(20, int(width_in * 72 / (0.6 * fontsize) * 0.92))
    return "\n".join(textwrap.fill(line, chars) or line for line in text.split("\n"))


def _value_label(src: Source, req: Request) -> str:
    v = src.vars[req.var]
    return (v.long_name or v.name) + (f" ({v.units})" if v.units else "")


def _map_data(src: Source, req: Request, cancelled: Callable[[], bool] | None = None) -> np.ndarray:
    """The plotted window as [y, x], reduced as the request asks."""
    v = src.vars[req.var]
    idx = req.index_dict()
    data = request_data(src, req, cancelled=cancelled)
    kept = [d for d in v.dims if isinstance(idx.get(d), tuple)]
    if kept.index(req.xdim) < kept.index(req.ydim):
        data = data.T
    return data


def _extend(lo: float, hi: float, data_lo: float, data_hi: float) -> str:
    below, above = data_lo < lo, data_hi > hi
    return "both" if below and above else "min" if below else "max" if above else "neither"


def _norm(req: Request, lo: float, hi: float):
    from matplotlib.colors import LogNorm, Normalize

    return LogNorm(vmin=lo, vmax=hi) if req.colour.log and lo > 0 else Normalize(vmin=lo, vmax=hi)


def _draw_map(fig, ax, src: Source, req: Request, data: np.ndarray, lo: float, hi: float, extend: str):
    """pcolormesh of a map or section with colour bar, coastlines and the TUI's orientation."""
    from matplotlib import colormaps as mpl_cmaps

    xd, yd = src.dims[req.xdim], src.dims[req.ydim]
    idx = req.index_dict()
    xa, xb = idx[req.xdim]
    ya, yb = idx[req.ydim]
    xc, yc = xd.values[xa:xb], yd.values[ya:yb]
    c = req.colour
    cmap = mpl_cmaps[c.cmap].with_extremes(bad=(0.35, 0.35, 0.35))
    mesh = ax.pcolormesh(_edges(xc), _edges(yc), np.ma.masked_invalid(data), cmap=cmap, norm=_norm(req, lo, hi),
                         shading="flat", rasterized=True)
    bar = fig.colorbar(mesh, ax=ax, label=_value_label(src, req), extend=extend)
    ax.set_xlabel(xd.axis_title())
    ax.set_ylabel(yd.axis_title())
    if req.options.coastlines and xd.is_lon and yd.is_lat:
        xe, ye = _edges(xc), _edges(yc)
        lon, lat = coastlines.load()
        lon_w, lat_w = coastlines.wrap(lon, lat, min(xe[0], xe[-1]), mirrored=yd.is_lat and yd.period is not None)
        ax.plot(lon_w, lat_w, color="black" if sum(mpl_cmaps[c.cmap](0.5)[:3]) > 1.5 else "white",
                linewidth=0.7)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
    if req.equal:
        ax.set_aspect(_cell_aspect(xd, yd))
    if xd.dates is not None:
        _time_ticks(ax.xaxis, xd, xc)
        fig.autofmt_xdate()
    if yd.dates is not None:
        _time_ticks(ax.yaxis, yd, yc, 8)
    for axis, dim in ((ax.xaxis, xd), (ax.yaxis, yd)):
        if dim.is_lat or dim.is_lon:
            _degree_locator(axis)
        if dim.period:
            _wrapped_labels(axis, dim)
    _orient(ax, xd, yd, req)
    return mesh, bar


def _line_values(src: Source, req: Request,
                 cancelled: Callable[[], bool] | None = None) -> tuple[np.ndarray, np.ndarray]:
    d = src.dims[req.xdim]
    a, b = req.index_dict()[req.xdim]
    return d.values[a:b], np.asarray(request_data(src, req, cancelled=cancelled), dtype=float).reshape(-1)


def _draw_line(fig, ax, src: Source, req: Request, value_range: tuple[float, float] | None,
               cancelled: Callable[[], bool] | None = None):
    """A profile (vertical coordinate up the side) or a line against the coordinate."""
    d = src.dims[req.xdim]
    coord, values = _line_values(src, req, cancelled)
    marker = "o" if len(values) <= 40 else None
    label = _value_label(src, req)
    if d.role == "z":
        (line,) = ax.plot(values, coord, marker=marker, markersize=3)
        ax.set_xlabel(label)
        ax.set_ylabel(d.axis_title())
        want_down = (d.positive == "down") != req.options.flip_y
        lo_c, hi_c = (np.nanmin(coord), np.nanmax(coord)) if len(coord) else (0, 1)
        ax.set_ylim((hi_c, lo_c) if want_down else (lo_c, hi_c))
        if value_range is not None:
            ax.set_xlim(*value_range)
    else:
        (line,) = ax.plot(coord, values, marker=marker, markersize=3)
        ax.set_xlabel(d.axis_title())
        ax.set_ylabel(label)
        if d.dates is not None:
            _time_ticks(ax.xaxis, d, coord)
            fig.autofmt_xdate()
        if d.is_lat or d.is_lon:
            _degree_locator(ax.xaxis)
        if d.period:
            _wrapped_labels(ax.xaxis, d)
        if value_range is not None:
            ax.set_ylim(*value_range)
    ax.grid(alpha=0.3)
    return line


def save_figure(
    src: Source,
    req: Request,
    path: str,
    title: str = "",
    size: tuple[float, float] = (8.0, 6.0),
    dpi: int = 150,
    vrange: tuple[float, float] | None = None,
) -> str:
    """Draw ``req`` with matplotlib and save it to ``path``. Returns the absolute path.

    ``vrange`` is the colour range for a map, or the value-axis range for a line.
    """
    path = _check_folder(path)
    fig, ax = _new_figure(size, dpi)
    if req.kind == "map":
        data = _map_data(src, req)
        lo, hi = vrange if vrange is not None else (float(np.nanmin(data)), float(np.nanmax(data)))
        finite = data[np.isfinite(data)]
        extend = _extend(lo, hi, float(finite.min()), float(finite.max())) if finite.size else "neither"
        _draw_map(fig, ax, src, req, data, lo, hi, extend)
    elif req.kind == "line":
        _draw_line(fig, ax, src, req, vrange)
    else:
        v = src.vars[req.var]
        value = float(np.asarray(request_data(src, req)).reshape(-1)[0])
        ax.text(0.5, 0.5, f"{v.name} = {value:.6g} {v.units}", ha="center", va="center", fontsize=16)
        ax.set_axis_off()
    if title:
        ax.set_title(_fit_title(title, size[0]), fontsize=10)
    fig.savefig(path, dpi=dpi)
    return os.path.abspath(path)


# --------------------------------------------------------------------------- animations

VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv")


def can_write_video() -> bool:
    """Is ffmpeg there for .mp4? (.gif always works.)"""
    import shutil

    return shutil.which("ffmpeg") is not None


def default_animation_filename(src: Source, req: Request, dim: str) -> str:
    """Like :func:`default_filename`, without the animated dimension's position."""
    base = default_filename(src, req, skip=dim)
    return base[: -len(".png")] + f"_{short_name(dim)}_anim" + (".mp4" if can_write_video() else ".gif")


def _at(req: Request, dim: str, i: int) -> Request:
    return replace(req, index=tuple((d, i if d == dim else ix) for d, ix in req.index))


def save_animation(
    src: Source,
    req: Request,
    dim: str,
    positions: list[int],
    path: str,
    title: str = "",
    size: tuple[float, float] = (8.0, 6.0),
    dpi: int = 100,
    fps: float = 5,
    vrange: tuple[float, float] | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> str:
    """Save ``req`` at each of ``positions`` along ``dim`` as a movie (.mp4, needs ffmpeg) or .gif.

    ``vrange`` fixes the colour (or value-axis) range for every frame; None fits each
    frame as the TUI's auto range does. Each frame's title gets that frame's header
    (view, time, level, what is reduced) under ``title``. Returns the absolute path.
    ``cancelled`` is asked between frames, and while a reduced frame is being read.
    """
    from matplotlib import animation

    from .frame import _header, colour_range

    path = _check_folder(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif ext in VIDEO_EXTENSIONS:
        if not can_write_video():
            raise OSError("saving video needs ffmpeg, which isn't installed: name the file .gif instead")
        writer = animation.FFMpegWriter(fps=fps, bitrate=-1)
    else:
        raise ValueError("name the file .mp4 or .gif")
    if not positions:
        raise ValueError("no frames to save")
    if req.kind not in ("map", "line"):
        raise ValueError("only maps, sections and line plots can be animated")

    v = src.vars[req.var]
    view = _view_name(src, req)
    others = [d for d in v.dims if d not in (req.xdim, req.ydim)]
    if req.kind == "line":
        others = [d for d in v.dims if d != req.xdim]
    frames = [_at(req, dim, i) for i in positions]

    def heading(r: Request) -> str:
        sub = _header(src, r, others, view)
        return _fit_title(f"{title}\n{sub}" if title else sub, size[0])

    fig, ax = _new_figure(size, dpi)
    if req.kind == "map":
        first = _map_data(src, frames[0], cancelled)
        if vrange is not None:
            # arrows on the colour bar if any frame goes beyond the fixed range
            dlo, dhi = np.inf, -np.inf
            for r in frames:
                d = _map_data(src, r, cancelled)
                if np.isfinite(d).any():
                    dlo, dhi = min(dlo, float(np.nanmin(d))), max(dhi, float(np.nanmax(d)))
            lo, hi = vrange
            extend = _extend(lo, hi, dlo, dhi) if np.isfinite(dlo) else "neither"
        else:
            lo, hi = colour_range(req.colour, first, None)
            extend = "neither"
        mesh, _ = _draw_map(fig, ax, src, req, first, lo, hi, extend)
    else:
        line = _draw_line(fig, ax, src, frames[0], vrange, cancelled)
    ax.set_title(heading(frames[0]), fontsize=10)
    fig.canvas.draw()
    fig.set_layout_engine("none")  # keep the layout still: colour bar labels change width

    if cancelled is not None and cancelled():
        raise Cancelled
    try:
        with writer.saving(fig, path, dpi):
            for k, r in enumerate(frames):
                # (after the first frame: a writer with no frames fails as it closes)
                if k and cancelled is not None and cancelled():
                    raise Cancelled
                if req.kind == "map":
                    data = _map_data(src, r, cancelled)
                    if vrange is None:
                        lo, hi = colour_range(req.colour, data, None)
                        mesh.norm.vmin, mesh.norm.vmax = lo, hi
                    mesh.set_array(np.ma.masked_invalid(data))
                else:
                    coord, values = _line_values(src, r, cancelled)
                    if src.dims[req.xdim].role == "z":
                        line.set_data(values, coord)
                    else:
                        line.set_data(coord, values)
                    if vrange is None:
                        ax.relim()
                        ax.autoscale_view(scalex=src.dims[req.xdim].role == "z",
                                          scaley=src.dims[req.xdim].role != "z")
                ax.set_title(heading(r), fontsize=10)
                writer.grab_frame()
                if progress is not None:
                    progress(k + 1, len(frames))
    except BaseException:
        if os.path.exists(path):
            os.remove(path)
        raise
    return os.path.abspath(path)


def _orient(ax, xd: Dim, yd: Dim, req: Request) -> None:
    """Match the TUI's orientation: coordinates increase right and up, pressure goes down."""
    if req.options.flip_x:
        ax.invert_xaxis()
    want_down = (yd.positive == "down") != req.options.flip_y
    if want_down:
        ax.invert_yaxis()
