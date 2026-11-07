"""The plot pane: draws the latest frame and turns keys and mouse into probe/zoom actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.segment import Segment
from textual import events
from textual.binding import Binding
from textual.strip import Strip
from textual.widget import Widget

from ..frame import Frame

if TYPE_CHECKING:
    from ..app import NctuiApp


class PlotView(Widget, can_focus=True):
    """Shows one plot. Arrows move the probe; click places it; drag zooms to a box."""

    BINDINGS = [
        Binding("left", "probe(-1, 0)", "probe", show=False),
        Binding("right", "probe(1, 0)", "probe", show=False),
        Binding("up", "probe(0, 1)", "probe", show=False),
        Binding("down", "probe(0, -1)", "probe", show=False),
        Binding("shift+left", "probe(-10, 0)", show=False),
        Binding("shift+right", "probe(10, 0)", show=False),
        Binding("shift+up", "probe(0, 10)", show=False),
        Binding("shift+down", "probe(0, -10)", show=False),
        Binding("enter", "app.plot_menu", "actions", key_display="enter"),
    ]

    DEFAULT_CSS = """
    PlotView {
        height: 1fr;
        width: 1fr;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame: Frame | None = None
        self._strips: list[Strip | None] = []
        self._drag_start: tuple[int, int] | None = None
        self._dragged = False

    @property
    def nctui(self) -> NctuiApp:
        return self.app  # type: ignore[return-value]

    # ------------------------------------------------------------------ drawing

    def show(self, frame: Frame) -> None:
        self.frame = frame
        self._strips = [None] * len(frame.lines)
        self.refresh()

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        frame = self.frame
        if frame is None or y >= len(frame.lines):
            return Strip.blank(width, self.rich_style)
        strip = self._strips[y]
        if strip is None or strip.cell_length != width:
            strip = self._strip(frame.lines[y], frame.width, width)
            self._strips[y] = strip
        return strip

    def _strip(self, line: list[Segment], line_width: int, width: int) -> Strip:
        """One frame line as a Strip of exactly ``width`` cells.

        Built once per frame line. Avoids ``Strip.apply_style``/``crop_extend``: those
        combine a style for every segment (thousands per frame) and, when the length
        already matches, cache the strip inside itself, a reference cycle that only
        the slow full garbage collection frees (see ``_textual_fixes``).
        """
        base = self.rich_style
        segs = []
        for seg in line:
            style = seg.style
            if style is None:
                segs.append(Segment(seg.text, base))
            elif style.bgcolor is None:
                segs.append(Segment(seg.text, base + style))
            else:
                segs.append(seg)  # image cells: both colours already set
        if line_width == width:
            return Strip(segs, width)
        return Strip(segs).adjust_cell_length(width, base)

    def on_resize(self, event: events.Resize) -> None:
        self.nctui.request_frame()

    # ------------------------------------------------------------------ keys

    def action_probe(self, dx: int, dy: int) -> None:
        if self.nctui.viewer.move_probe(dx, dy):
            self.nctui.view_changed()

    # ------------------------------------------------------------------ mouse

    def _pixel_at(self, x: int, y: int) -> tuple[int, int] | None:
        """Image pixel under a cell (top half of the cell), or None outside the image."""
        f = self.frame
        if f is None or not f.maps:
            return None
        req = f.request
        if req.kind == "map":
            xm, ym = f.maps.get(req.xdim), f.maps.get(req.ydim)
            if xm is None or ym is None:
                return None
            px = x - f.image_col
            py = 2 * (y - f.image_row) - f.pad_top
            if 0 <= px < xm.out and -1 <= py < ym.out:
                return px, max(py, 0)
            return None
        if req.kind == "line" and f.line_fig is not None:
            lf = f.line_fig
            cx, cy = x - lf.plot_col, y - lf.plot_row
            if 0 <= cx < lf.cols and 0 <= cy < lf.rows:
                return cx, cy
        return None

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        self.focus()
        p = self._pixel_at(event.x, event.y)
        if p is None:
            return
        self._drag_start = p
        self._dragged = False
        self.capture_mouse()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._drag_start is None or self.frame is None or self.frame.request.kind != "map":
            return
        f = self.frame
        xm, ym = f.maps[f.request.xdim], f.maps[f.request.ydim]
        px = min(max(event.x - f.image_col, 0), xm.out - 1)
        py = min(max(2 * (event.y - f.image_row) - f.pad_top, 0), ym.out - 1)
        x0, y0 = self._drag_start
        if abs(px - x0) + abs(py - y0) >= 2:
            self._dragged = True
            self.nctui.viewer.selection = (x0, y0, px, py)
            self.nctui.request_frame()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._drag_start is None:
            return
        self.release_mouse()
        start, self._drag_start = self._drag_start, None
        viewer = self.nctui.viewer
        f = self.frame
        if self._dragged and viewer.selection is not None:
            x0, y0, x1, y1 = viewer.selection
            viewer.selection = None
            viewer.zoom_to_pixels(x0, y0, x1, y1)
            self.nctui.view_changed()
            return
        viewer.selection = None
        if f is None:
            return
        if f.request.kind == "map":
            if viewer.set_probe_pixel(*start):
                self.nctui.view_changed()
        elif f.request.kind == "line" and f.line_fig is not None:
            self.nctui.click_line(start[0], start[1])
