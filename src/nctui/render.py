"""Turn data into terminal graphics: half-block images with axes, and braille line plots.

Everything here is a pure function that produces lists of Rich ``Segment`` lines,
so the same code draws the TUI and ``nctui plot`` on stdout. No Textual imports.

Images use the upper-half-block character ``▀``: its foreground colour is the top
pixel and its background colour the bottom pixel, so each character cell holds two
roughly square pixels.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from rich.color import Color
from rich.segment import Segment
from rich.style import Style

from . import colormaps
from .data import format_number

Line = list[Segment]

# palette layout: 0..255 colour map, then special colours
NAN, BLACK, WHITE, ACCENT, COAST, BG = 256, 257, 258, 259, 260, -1
ACCENT_RGB = (255, 95, 215)
AXIS_STYLE = Style(color="grey62")
LABEL_STYLE = Style(color="grey78")
TITLE_STYLE = Style(color="grey85", bold=True)
ACCENT_STYLE = Style(color=Color.from_rgb(*ACCENT_RGB), bold=True)
LINE_RGB = (97, 175, 239)
LINE_STYLE = Style(color=Color.from_rgb(*LINE_RGB))

# Space a figure needs around the image (in character cells).
CBAR_COLS = 2
CBAR_LABEL_COLS = 9
RIGHT_COLS = 2 + CBAR_COLS + 1 + CBAR_LABEL_COLS  # gap, bar, gap, labels
TOP_ROWS = 1  # axis titles
BOTTOM_ROWS = 3  # x axis line, tick labels, x title


# --------------------------------------------------------------------------- palette & styles


@lru_cache(maxsize=64)
def palette(cmap: str) -> np.ndarray:
    pal = np.zeros((261, 3), dtype=np.uint8)
    pal[:256] = colormaps.lut(cmap)
    pal[NAN] = colormaps.NAN_RGB
    pal[BLACK] = (0, 0, 0)
    pal[WHITE] = (255, 255, 255)
    pal[ACCENT] = ACCENT_RGB
    pal[COAST] = colormaps.contrast_rgb(cmap)
    return pal


@lru_cache(maxsize=64)
def _luminance(cmap: str) -> np.ndarray:
    t = palette(cmap).astype(float) / 255
    return 0.2126 * t[:, 0] + 0.7152 * t[:, 1] + 0.0722 * t[:, 2]


@lru_cache(maxsize=300_000)
def _cell_style(cmap: str, top: int, bottom: int) -> Style:
    pal = palette(cmap)
    fg = Color.from_rgb(*pal[top]) if top != BG else None
    bg = Color.from_rgb(*pal[bottom]) if bottom != BG else None
    return Style(color=fg, bgcolor=bg)


def colour_indices(
    data: np.ndarray, vmin: float, vmax: float, log: bool = False
) -> np.ndarray:
    """Map data onto palette indices 0..255, with NaN → ``NAN``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        if log:
            lo = math.log10(vmin) if vmin > 0 else -1.0
            hi = math.log10(vmax) if vmax > 0 else lo + 1
            vals = np.log10(np.where(data > 0, data, np.nan))
            vals = np.where(np.isfinite(data) & (data <= 0), lo, vals)
        else:
            lo, hi, vals = vmin, vmax, data
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            frac = np.full(data.shape, 0.5)
        else:
            frac = (vals - lo) / span
        idx = np.clip(np.floor(frac * 256), 0, 255)
    out = np.where(np.isfinite(data), idx, NAN).astype(np.int16)
    return out


def halfblock_rows(idx: np.ndarray, cmap: str) -> list[Line]:
    """Rows of ``▀`` cells for a palette-index image (row 0 at the top)."""
    h, w = idx.shape
    if h % 2:
        idx = np.vstack([np.full((1, w), BG, dtype=idx.dtype), idx])
    rows: list[Line] = []
    for r in range(0, idx.shape[0], 2):
        top, bot = idx[r], idx[r + 1]
        segs: Line = []
        run_start = 0
        for c in range(1, w + 1):
            if c == w or top[c] != top[run_start] or bot[c] != bot[run_start]:
                t, b = int(top[run_start]), int(bot[run_start])
                n = c - run_start
                if t == BG and b == BG:
                    segs.append(Segment(" " * n))
                elif t == BG:
                    segs.append(Segment("▄" * n, _cell_style(cmap, b, BG)))
                elif t == b:  # one colour: a space needs only the background (fewer bytes to send)
                    segs.append(Segment(" " * n, _cell_style(cmap, BG, b)))
                else:
                    segs.append(Segment("▀" * n, _cell_style(cmap, t, b)))
                run_start = c
        rows.append(segs)
    return rows


# --------------------------------------------------------------------------- ticks


def nice_step(span: float, max_ticks: int) -> float:
    if span <= 0 or not np.isfinite(span) or max_ticks < 1:
        return 1.0
    raw = span / max_ticks
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def nice_ticks(lo: float, hi: float, max_ticks: int) -> list[float]:
    """Round-numbered tick values within [lo, hi]."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return []
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        return [lo]
    step = nice_step(hi - lo, max_ticks)
    digits = max(0, 1 - int(math.floor(math.log10(step))))
    k0 = math.ceil(lo / step - 1e-9)
    k1 = math.floor(hi / step + 1e-9)
    if k1 - k0 > 1000:
        return []
    return [round(k * step, digits) + 0.0 for k in range(k0, k1 + 1)]


# steps for latitude and longitude ticks: each divides 90, so the equator and poles get ticks
DEGREE_STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 45, 90)


def degree_ticks(lo: float, hi: float, max_ticks: int) -> list[float]:
    """Tick values for a latitude or longitude axis, at steps that land on 0, ±90 and ±180."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return []
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    step = next((st for st in DEGREE_STEPS if span / st <= max(1, max_ticks)), None)
    if span <= 0 or step is None or step == DEGREE_STEPS[0] and span / step < 2:
        return nice_ticks(lo, hi, max_ticks)
    k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    return [round(k * step, 6) + 0.0 for k in range(k0, k1 + 1)]


def format_ticks(ticks: list[float]) -> list[str]:
    """Labels for evenly spaced tick values, with just enough decimals to tell them apart."""
    if not ticks:
        return []
    if len(ticks) < 2:
        return [format_number(t, 4) for t in ticks]
    step = min(abs(b - a) for a, b in zip(ticks, ticks[1:], strict=False) if b != a) if len(set(ticks)) > 1 else 1.0
    big = max(abs(t) for t in ticks)
    tol = step * 1e-6
    if big >= 1e6 or (0 < big < 1e-3):
        for d in range(0, 8):
            if all(abs(float(f"{t:.{d}e}") - t) <= tol for t in ticks):
                break
        return ["0" if t == 0 else _compact_exp(f"{t:.{d}e}") for t in ticks]
    for d in range(0, 10):
        if all(abs(round(t, d) - t) <= tol for t in ticks):
            break
    return [f"{t:.{d}f}" for t in ticks]


def _compact_exp(text: str) -> str:
    mant, exp = text.split("e")
    return f"{mant}e{int(exp):+03d}"


@dataclass
class AxisSpec:
    """One axis of a figure."""

    title: str
    centres: np.ndarray  # coordinate value at the centre of each output pixel, screen order
    label: Callable[[float], str] | None = None  # None: numeric ticks (format_ticks)
    probe: int | None = None  # pixel (screen order) of the probe, if any
    integer: bool = False  # ticks only at whole numbers (index or time axes)
    tick_values: Callable[[float, float, int], list[float]] | None = None  # None: nice_ticks
    wrap: Callable[[float], float] | None = None  # how a (centred, periodic) value reads


def _tick_positions(axis: AxisSpec, max_ticks: int) -> list[tuple[float, str]]:
    """(pixel position, label) for round-valued ticks along an axis."""
    c = np.asarray(axis.centres, dtype=float)
    n = len(c)
    if n == 0:
        return []
    finite = np.isfinite(c)
    if not finite.any():
        return []
    lo, hi = float(np.nanmin(c)), float(np.nanmax(c))
    if n == 1 or lo == hi:
        return [(0.0 if n == 1 else (n - 1) / 2, _labels(axis, [lo])[0])]
    if axis.tick_values is not None:
        ticks = axis.tick_values(lo, hi, max_ticks)
    else:
        ticks = nice_ticks(lo, hi, max_ticks)
    if axis.integer:
        ticks = sorted({float(round(t)) for t in ticks})
    increasing = c[-1] >= c[0]
    pix = np.arange(n, dtype=float)
    xs, ps = (c, pix) if increasing else (c[::-1], pix[::-1])
    return [(float(np.interp(t, xs, ps)), label) for t, label in zip(ticks, _labels(axis, ticks), strict=True)]


def _pixel(p: float) -> int:
    """The pixel a fractional tick position falls in; halves round down, like the probe marker."""
    return int(math.floor(p + 0.5 - 1e-9))


def max_x_ticks(axis: AxisSpec, cols: int) -> int:
    """How many ticks fit along a horizontal axis ``cols`` cells long."""
    if axis.label is None or len(axis.centres) == 0:
        return max(2, cols // 10)
    width = len(axis.label(float(axis.centres[0])))
    return max(1, cols // (width + 3))


def _labels(axis: AxisSpec, ticks: list[float]) -> list[str]:
    if axis.label is None:
        return format_ticks([axis.wrap(t) for t in ticks] if axis.wrap else ticks)
    return [axis.label(t) for t in ticks]


def y_label_width(axis: AxisSpec) -> int:
    c = np.asarray(axis.centres, dtype=float)
    if len(c) == 0 or not np.isfinite(c).any():
        return 1
    ticks = nice_ticks(float(np.nanmin(c)), float(np.nanmax(c)), 12) or [float(np.nanmin(c))]
    labels = _labels(axis, ticks)
    return max(1, min(10, max(len(s) for s in labels)))


def image_area(width: int, height: int, ylab_width: int) -> tuple[int, int]:
    """Pixels available for the image in a ``width`` × ``height`` cell figure."""
    cols = width - ylab_width - 1 - RIGHT_COLS
    rows = height - TOP_ROWS - BOTTOM_ROWS
    return max(0, cols), max(0, 2 * rows)


# --------------------------------------------------------------------------- figure pieces


def _pad(line: Line, width: int) -> Line:
    n = sum(s.cell_length for s in line)
    if n < width:
        line = line + [Segment(" " * (width - n))]
    return line


def _place(text: str, width: int, positions: list[tuple[int, str]]) -> str:
    """Place labels centred at column positions, skipping any that would overlap."""
    buf = [" "] * width
    last_end = -2
    for col, label in sorted(positions):
        start = int(col - len(label) // 2)
        start = max(0, min(width - len(label), start))
        if start <= last_end + 1:
            continue
        for i, ch in enumerate(label):
            if 0 <= start + i < width:
                buf[start + i] = ch
        last_end = start + len(label) - 1
    return "".join(buf)


@dataclass
class ColourBar:
    cmap: str
    vmin: float
    vmax: float
    log: bool = False
    units: str = ""


def colourbar_rows(bar: ColourBar, nrows: int) -> list[Line]:
    """Rows of the vertical colour bar with labels (``CBAR_COLS + 1 + labels``)."""
    rows: list[Line] = []
    if nrows <= 0:
        return rows
    npix = 2 * nrows
    frac = 1.0 - (np.arange(npix) + 0.5) / npix  # top = 1
    idx = np.clip(np.floor(frac * 256), 0, 255).astype(np.int16)
    # tick labels
    labels: dict[int, str] = {}
    if bar.log and bar.vmin > 0 and bar.vmax > bar.vmin:
        lo, hi = math.log10(bar.vmin), math.log10(bar.vmax)
        ticks = [t for t in nice_ticks(lo, hi, max(2, nrows // 3)) if float(t).is_integer()] or [lo, hi]
        positions = [((hi - t) / (hi - lo), f"1e{int(t)}" if float(t).is_integer() else format_number(10 ** t, 3))
                     for t in ticks]
    else:
        span = bar.vmax - bar.vmin
        ticks = nice_ticks(bar.vmin, bar.vmax, max(2, nrows // 3))
        if span > 0 and np.isfinite(span):
            positions = [((bar.vmax - t) / span, label) for t, label in zip(ticks, format_ticks(ticks), strict=True)]
        else:
            positions = [(0.5, format_number(bar.vmin, 4))]
    for f, text in positions:
        r = int(np.clip(math.floor(f * nrows - 1e-9) if f > 0 else 0, 0, nrows - 1))
        if all(abs(r - other) > 1 for other in labels):
            labels[r] = text[:CBAR_LABEL_COLS]
    for r in range(nrows):
        t, b = int(idx[2 * r]), int(idx[2 * r + 1])
        line: Line = [Segment("▀" * CBAR_COLS, _cell_style(bar.cmap, t, b))]
        label = labels.get(r)
        if label is not None:
            line.append(Segment("╴" + label.ljust(CBAR_LABEL_COLS), LABEL_STYLE))
        else:
            line.append(Segment(" " * (1 + CBAR_LABEL_COLS)))
        rows.append(line)
    return rows


@dataclass
class MapFigure:
    """A half-block image with axes and a colour bar, ready to draw."""

    lines: list[Line]
    image_col: int  # column of the image's left edge
    image_row: int  # row of the image's first cell row
    pad_top: int  # 1 if the image has an odd pixel height (blank half-cell on top)
    width: int = 0
    height: int = 0


def map_figure(
    idx: np.ndarray,
    cmap: str,
    xaxis: AxisSpec,
    yaxis: AxisSpec,
    bar: ColourBar,
    width: int,
    height: int,
) -> MapFigure:
    """Lay out a palette-index image (already fitted to the pixel grid) as a figure."""
    out_h, out_w = idx.shape
    ylw = y_label_width(yaxis)
    img_rows = (out_h + 1) // 2
    pad_top = out_h % 2
    image_col = ylw + 1
    image_row = TOP_ROWS
    bar_rows = max(img_rows, min(6, height - TOP_ROWS - BOTTOM_ROWS))
    body_rows = max(img_rows, bar_rows)

    lines: list[Line] = []
    # title row: y axis title on the left, colour bar units over the bar
    left_title = yaxis.title[: max(0, ylw + 1 + out_w)]
    units = bar.units[: CBAR_COLS + 1 + CBAR_LABEL_COLS]
    title_gap = max(1, image_col + out_w + 2 - len(left_title))
    lines.append([Segment(left_title, TITLE_STYLE), Segment(" " * title_gap), Segment(units, LABEL_STYLE)])

    # y ticks per cell row
    y_ticks: dict[int, str] = {}
    for p, label in _tick_positions(yaxis, max(2, img_rows // 3)):
        r = (_pixel(p) + pad_top) // 2
        if 0 <= r < img_rows and all(abs(r - o) > 1 for o in y_ticks):
            y_ticks[r] = label
    probe_row = None
    if yaxis.probe is not None:
        probe_row = (yaxis.probe + pad_top) // 2

    img_lines = halfblock_rows(idx, cmap)
    cbar = colourbar_rows(bar, bar_rows)
    for r in range(body_rows):
        line: Line = []
        if r < img_rows:
            label = y_ticks.get(r, "")
            line.append(Segment(label.rjust(ylw)[:ylw], LABEL_STYLE))
            if r == probe_row:
                line.append(Segment("▶", ACCENT_STYLE))
            else:
                line.append(Segment("┤" if r in y_ticks else "│", AXIS_STYLE))
            line.extend(img_lines[r])
        else:
            line.append(Segment(" " * (ylw + 1 + out_w)))
        line.append(Segment("  "))
        if r < len(cbar):
            line.extend(cbar[r])
        lines.append(line)

    # x axis line with ticks, then labels, then title
    x_ticks = _tick_positions(xaxis, max_x_ticks(xaxis, out_w))
    axis_chars = ["─"] * out_w
    tick_labels: list[tuple[int, str]] = []
    for p, label in x_ticks:
        c = _pixel(p)
        if 0 <= c < out_w:
            axis_chars[c] = "┬"
            tick_labels.append((c, label))
    probe_col = xaxis.probe if xaxis.probe is not None and 0 <= xaxis.probe < out_w else None
    axis_line: Line = [Segment(" " * ylw), Segment("└", AXIS_STYLE)]
    if probe_col is not None:
        axis_line += [
            Segment("".join(axis_chars[:probe_col]), AXIS_STYLE),
            Segment("▲", ACCENT_STYLE),
            Segment("".join(axis_chars[probe_col + 1:]), AXIS_STYLE),
        ]
    else:
        axis_line.append(Segment("".join(axis_chars), AXIS_STYLE))
    lines.append(axis_line)
    lines.append([Segment(" " * (ylw + 1)), Segment(_place("", out_w + 8, tick_labels), LABEL_STYLE)])
    title = xaxis.title[: out_w + ylw]
    start = ylw + 1 + max(0, (out_w - len(title)) // 2)
    lines.append([Segment(" " * start), Segment(title, TITLE_STYLE)])
    lines = [_pad(line, width) for line in lines[:height]]
    return MapFigure(lines=lines, image_col=image_col, image_row=image_row, pad_top=pad_top,
                     width=width, height=len(lines))


# --------------------------------------------------------------------------- line plots

_BRAILLE = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


@dataclass
class LineFigure:
    lines: list[Line]
    plot_col: int  # first column of the plotting area
    plot_row: int
    cols: int
    rows: int
    vertical: bool  # coordinate runs up the y axis (profile)
    coord_dots: int = 0  # dots along the coordinate direction
    extra: dict = field(default_factory=dict)


def line_figure(
    coord: np.ndarray,
    values: np.ndarray,
    width: int,
    height: int,
    coord_axis: AxisSpec,
    value_title: str,
    vertical: bool = False,
    coord_up: bool = True,
    marker: int | None = None,
    value_range: tuple[float, float] | None = None,
) -> LineFigure:
    """A braille line plot of ``values`` against ``coord``.

    ``vertical`` puts the coordinate on the y axis (a profile); ``coord_up`` says
    whether the coordinate increases upwards in that case. ``marker`` is an index
    into the arrays to highlight.
    """
    coord = np.asarray(coord, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if value_range is not None:
        vlo, vhi = value_range
    elif finite.any():
        vlo, vhi = float(np.nanmin(values)), float(np.nanmax(values))
    else:
        vlo, vhi = 0.0, 1.0
    if vhi == vlo:
        pad = abs(vlo) * 0.05 or 0.5
        vlo, vhi = vlo - pad, vhi + pad
    else:
        pad = (vhi - vlo) * 0.05
        vlo, vhi = vlo - pad, vhi + pad

    value_axis = AxisSpec(title=value_title, centres=np.linspace(vlo, vhi, 50))
    clo, chi = (float(np.nanmin(coord)), float(np.nanmax(coord))) if len(coord) else (0.0, 1.0)
    if chi == clo:
        clo, chi = clo - 0.5, chi + 0.5

    if vertical:
        left_axis = replace(coord_axis, centres=np.linspace(chi, clo, 50) if coord_up else np.linspace(clo, chi, 50),
                            probe=None)
        bottom_axis = value_axis
    else:
        left_axis = AxisSpec(value_title, np.linspace(vhi, vlo, 50))
        bottom_axis = replace(coord_axis, centres=np.linspace(clo, chi, 50), probe=None)
    ylw = y_label_width(left_axis)
    cols = max(1, width - ylw - 1 - 2)
    rows = max(1, height - TOP_ROWS - BOTTOM_ROWS)
    dw, dh = 2 * cols, 4 * rows

    def to_dot_x(v):
        lo, hi = (vlo, vhi) if vertical else (clo, chi)
        return (v - lo) / (hi - lo) * (dw - 1)

    def to_dot_y(v):
        if vertical:
            f = (v - clo) / (chi - clo)
            return (1 - f) * (dh - 1) if coord_up else f * (dh - 1)
        return (vhi - v) / (vhi - vlo) * (dh - 1)

    if vertical:
        dx, dy = to_dot_x(values), to_dot_y(coord)
    else:
        dx, dy = to_dot_x(coord), to_dot_y(values)

    dots = np.zeros((rows, cols), dtype=np.int32)

    def plot_dot(x: float, y: float) -> None:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < dw and 0 <= yi < dh:
            dots[yi // 4, xi // 2] |= _BRAILLE[yi % 4][xi % 2]

    ok = np.isfinite(dx) & np.isfinite(dy)
    for i in range(len(dx)):
        if not ok[i]:
            continue
        if i + 1 < len(dx) and ok[i + 1]:
            steps = int(max(abs(dx[i + 1] - dx[i]), abs(dy[i + 1] - dy[i]))) + 1
            for t in np.linspace(0, 1, steps + 1):
                plot_dot(dx[i] + (dx[i + 1] - dx[i]) * t, dy[i] + (dy[i + 1] - dy[i]) * t)
        else:
            plot_dot(dx[i], dy[i])

    # overlay: data points (when sparse) and the marker
    overlay: dict[tuple[int, int], tuple[str, Style]] = {}
    n_ok = int(ok.sum())
    if 0 < n_ok <= max(2, min(cols, 4 * rows) // 3):
        for i in np.flatnonzero(ok):
            r, c = int(round(dy[i])) // 4, int(round(dx[i])) // 2
            if 0 <= r < rows and 0 <= c < cols:
                overlay[(r, c)] = ("•", LINE_STYLE)
    marker_cell = None
    if marker is not None and 0 <= marker < len(dx):
        mx = dx[marker] if np.isfinite(dx[marker]) else None
        my = dy[marker] if np.isfinite(dy[marker]) else None
        if vertical and my is not None:
            mr = int(np.clip(round(my) // 4, 0, rows - 1))
            for c in range(cols):
                if dots[mr, c] == 0 and (mr, c) not in overlay:
                    overlay[(mr, c)] = ("┄", AXIS_STYLE)
            if mx is not None:
                marker_cell = (mr, int(np.clip(round(mx) // 2, 0, cols - 1)))
        elif not vertical and mx is not None:
            mc = int(np.clip(round(mx) // 2, 0, cols - 1))
            for r in range(rows):
                if dots[r, mc] == 0 and (r, mc) not in overlay:
                    overlay[(r, mc)] = ("┊", AXIS_STYLE)
            if my is not None:
                marker_cell = (int(np.clip(round(my) // 4, 0, rows - 1)), mc)
        if marker_cell is not None:
            overlay[marker_cell] = ("●", ACCENT_STYLE)

    lines: list[Line] = [[Segment(left_axis.title[: width], TITLE_STYLE)]]
    left_axis.centres = np.linspace(left_axis.centres[0], left_axis.centres[-1], dh)
    y_ticks: dict[int, str] = {}
    for p, label in _tick_positions(left_axis, max(2, rows // 3)):
        r = int(round(p)) // 4
        if 0 <= r < rows and all(abs(r - o) > 1 for o in y_ticks):
            y_ticks[r] = label
    for r in range(rows):
        line: Line = [Segment(y_ticks.get(r, "").rjust(ylw)[:ylw], LABEL_STYLE),
                      Segment("┤" if r in y_ticks else "│", AXIS_STYLE)]
        run = []
        for c in range(cols):
            if (r, c) in overlay:
                if run:
                    line.append(Segment("".join(run), LINE_STYLE))
                    run = []
                ch, st = overlay[(r, c)]
                line.append(Segment(ch, st))
            else:
                run.append(chr(0x2800 + int(dots[r, c])))
        if run:
            line.append(Segment("".join(run), LINE_STYLE))
        lines.append(line)

    bottom_axis.centres = np.linspace(bottom_axis.centres[0], bottom_axis.centres[-1], cols)
    ticks = [(_pixel(p), label) for p, label in _tick_positions(bottom_axis, max_x_ticks(bottom_axis, cols))]
    axis_chars = ["─"] * cols
    for c, _ in ticks:
        if 0 <= c < cols:
            axis_chars[c] = "┬"
    lines.append([Segment(" " * ylw), Segment("└" + "".join(axis_chars), AXIS_STYLE)])
    lines.append([Segment(" " * (ylw + 1)), Segment(_place("", cols + 6, ticks), LABEL_STYLE)])
    title = bottom_axis.title[:cols]
    lines.append([Segment(" " * (ylw + 1 + max(0, (cols - len(title)) // 2))), Segment(title, TITLE_STYLE)])
    lines = [_pad(line, width) for line in lines[:height]]
    return LineFigure(lines=lines, plot_col=ylw + 1, plot_row=TOP_ROWS, cols=cols, rows=rows,
                      vertical=vertical, coord_dots=dh if vertical else dw)


def text_block(message: str, width: int, height: int, style: Style | None = None) -> list[Line]:
    """A message centred in an otherwise blank area."""
    import textwrap

    lines: list[Line] = []
    msg_lines = [w for part in message.splitlines() for w in (textwrap.wrap(part, max(width, 1)) or [""])] or [""]
    top = max(0, (height - len(msg_lines)) // 2)
    for r in range(height):
        i = r - top
        if 0 <= i < len(msg_lines):
            text = msg_lines[i][:width]
            left = max(0, (width - len(text)) // 2)
            lines.append([Segment(" " * left), Segment(text, style or LABEL_STYLE),
                          Segment(" " * max(0, width - left - len(text)))])
        else:
            lines.append([Segment(" " * width)])
    return lines
