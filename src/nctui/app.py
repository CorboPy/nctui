"""The nctui Textual application."""

from __future__ import annotations

import gc
import time
from collections import deque

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Footer, Input, Static
from textual.worker import get_current_worker

from . import export
from ._textual_fixes import patch_strip_divide, patch_strip_render
from .data import TIME, Source, Z
from .frame import Frame, build_frame, short_name
from .screens.colours import ColoursScreen
from .screens.goto import GoToScreen
from .screens.info import HelpScreen, InfoScreen
from .screens.menu import ActionMenu
from .screens.save import AnimOptions, SaveOptions, SaveScreen
from .screens.settings import PlotSettingsScreen
from .state import Viewer
from .widgets.dimensions import DimensionPanel
from .widgets.plot import PlotView
from .widgets.variables import VariableTree

FPS_STEPS = [1, 2, 3, 5, 8, 12, 20, 30]
READ_AHEAD = 4  # frames of data read ahead while animating

patch_strip_divide()
patch_strip_render()


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


class NctuiApp(App):
    TITLE = "nctui"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    #topbar {
        height: 1;
        background: $primary-background;
    }
    #topbar-left { width: 1fr; padding: 0 1; }
    #topbar-right { width: auto; padding: 0 1; color: $text-muted; }
    #body { height: 1fr; }
    #sidebar { width: 40; }
    #sidebar.hidden { display: none; }
    #search {
        display: none;
        height: 3;
        border: round $accent;
    }
    #search.visible { display: block; }
    VariableTree {
        height: 1fr;
        border: round $panel-lighten-2;
        border-title-color: $text;
        border-subtitle-color: $text-muted;
        background: $background;
        padding: 0;
    }
    VariableTree:focus { border: round $accent; }
    DimensionPanel {
        border: round $panel-lighten-2;
        border-title-color: $text;
        border-subtitle-color: $text-muted;
        max-height: 60%;
    }
    DimensionPanel:focus { border: round $accent; }
    #main {
        width: 1fr;
        border: round $panel-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
        border-subtitle-color: $text-muted;
        padding: 0 1;
    }
    #main:focus-within { border: round $accent; }
    #header { height: 1; text-style: bold; }
    #stats, #status { height: 1; color: $text-muted; }
    """

    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("question_mark", "help", "help", key_display="?"),
        Binding("slash", "search", "find", key_display="/", show=False),
        Binding("left_square_bracket", "step_a(-1)", "time", key_display="[ ]"),
        Binding("right_square_bracket", "step_a(1)", show=False),
        Binding("left_curly_bracket", "step_b(-1)", "level", key_display="{ }"),
        Binding("right_curly_bracket", "step_b(1)", show=False),
        Binding("space", "play", "play"),
        Binding("less_than_sign", "speed(-1)", show=False),
        Binding("greater_than_sign", "speed(1)", show=False),
        Binding("v", "cycle_view", "view"),
        Binding("t", "swap_axes", show=False),
        Binding("plus", "zoom(2)", show=False),
        Binding("equals_sign", "zoom(2)", show=False),
        Binding("minus", "zoom(0.5)", show=False),
        Binding("1", "every_point", show=False),
        Binding("0", "fit", show=False),
        Binding("f", "full_width", show=False),
        Binding("c", "colours", "colours"),
        Binding("m", "next_cmap", show=False),
        Binding("r", "lock_range", show=False),
        Binding("p", "plot_settings", "settings"),
        Binding("o", "coastlines", show=False),
        Binding("g", "goto", show=False),
        Binding("e", "save", "save"),
        Binding("i", "info", show=False),
        Binding("escape", "back", show=False),
    ]

    def __init__(self, source: Source, start_var: str | None = None):
        super().__init__()
        self.src = source
        self.viewer = Viewer(source)
        self.start_var = start_var
        self.last_frame: Frame | None = None
        self._pending: tuple | None = None
        self._building = False
        self._select_timer: Timer | None = None
        self._anim_timer: Timer | None = None
        self._play_error: str | None = None  # why space did nothing (shown in red for a moment)
        self._shown: deque[float] = deque(maxlen=60)  # when animation frames reached the screen
        self._topbar_extra = ""
        self._topbar_at = 0.0
        self._anim_dim: str | None = None
        self._fps_i = FPS_STEPS.index(5)
        self._anim_started = 0.0
        self._scan_shown = -1
        self._scan_key: tuple | None = None
        self._frozen = False

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static(id="topbar-left")
            yield Static("? help · q quit", id="topbar-right")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Input(placeholder="search name, long_name, standard_name", id="search")
                yield VariableTree(self.src, id="vars")
                yield DimensionPanel(id="dims")
            with Vertical(id="main"):
                yield Static(id="header")
                yield PlotView(id="plot")
                yield Static(id="stats")
                yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one(VariableTree)
        tree.border_title = "Variables"
        tree.border_subtitle = "/ search"
        dims = self.query_one(DimensionPanel)
        dims.border_title = "Dimensions"
        dims.border_subtitle = "←→ step · x y axes"
        tree.rebuild()
        start = self.start_var if self.start_var in self.src.vars else None
        start = start or tree.first_variable()
        self._update_topbar()
        if start is not None:
            self.viewer.select_var(start)
            self.call_after_refresh(tree.highlight, start)
        tree.focus()
        self.update_chrome()
        self.call_after_refresh(self.request_frame)
        for w in self.src.warnings:
            self.notify(w, severity="warning", timeout=8)

    def _update_topbar(self, extra: str = "") -> None:
        self._topbar_extra = extra
        self._topbar_at = time.monotonic()
        text = Text()
        text.append("nctui ", style="bold")
        text.append(self.src.title)
        text.append(f" · {human_size(self.src.nbytes_on_disk)}", style="dim")
        if self._anim_timer is not None:
            fps = FPS_STEPS[self._fps_i]
            actual = self._shown_fps()
            if actual is not None and actual < 0.85 * fps:  # the terminal can't draw them all
                text.append(f"   ▶ playing {self._anim_dim} · {actual:.0f} of {fps} fps", style="bold yellow")
            else:
                text.append(f"   ▶ playing {self._anim_dim} at {fps} fps", style="bold green")
        elif self._play_error:
            text.append(f"   ■ {self._play_error}", style="bold red")
        if extra:
            text.append(f"   {extra}", style="yellow")
        self.query_one("#topbar-left", Static).update(text)

    # ------------------------------------------------------------------ state → screen

    def update_chrome(self) -> None:
        viewer = self.viewer
        main = self.query_one("#main")
        if viewer.var is not None:
            v = viewer.v
            parts = [v.name]
            if v.long_name and v.long_name != v.name:
                parts.append(v.long_name)
            if v.units:
                parts.append(v.units)
            main.border_title = " · ".join(parts)
            main.border_subtitle = "enter: profile / series here · esc: back" if viewer.kind == "map" else (
                "esc: back to map" if viewer.stack else "")
        self.query_one(DimensionPanel).refresh(layout=True)

    def view_changed(self) -> None:
        dim = self._anim_dim
        if self._anim_timer is not None and dim not in self.viewer.free_dims():
            # the animated dimension went onto the plot (v, t, x/y) or the variable lacks it
            self._stop_animation()
            self._cant_play("stopped: view changed")
        self.update_chrome()
        self.request_frame()

    def request_frame(self) -> None:
        req = self.viewer.request()
        if req is None:
            return
        plot = self.query_one(PlotView)
        w, h = plot.size.width, plot.size.height
        if w <= 0 or h <= 0:
            return
        prefetch = []
        here = dict(req.index).get(self._anim_dim)
        if self._anim_timer is not None and isinstance(here, int):
            # read the next few frames' data while this one is drawn (slow disks)
            size = self.src.dims[self._anim_dim].size
            for k in range(1, min(READ_AHEAD, size - 1) + 1):
                idx = dict(req.index)
                idx[self._anim_dim] = (here + k) % size
                prefetch.append(idx)
        self._pending = (req, w, h, prefetch)
        if not self._building:
            self._start_next()
        if req.colour.mode == "anim" and req.anim_range is None:
            self._start_anim_range_scan()

    def _start_next(self) -> None:
        if self._pending is None:
            return
        job, self._pending = self._pending, None
        self._building = True
        self._build(*job)

    @work(thread=True, group="frame", exit_on_error=False)
    def _build(self, req, w, h, prefetch) -> None:
        frame = build_frame(self.src, req, w, h)
        self.call_from_thread(self._frame_done, frame)
        for idx in prefetch:
            try:
                self.src.read(req.var, idx)  # cached reads return at once
            except Exception:  # noqa: BLE001 - prefetching is best effort
                return

    def _frame_done(self, frame: Frame) -> None:
        self._building = False
        self.last_frame = frame
        if not self._frozen:
            # everything alive after the first frame (Textual, the file's metadata) lives
            # for the whole session: keep it out of the garbage collector's full scans
            self._frozen = True
            gc.freeze()
        viewer = self.viewer
        if frame.request.var == viewer.var:
            viewer.maps = frame.maps
            viewer.area = frame.area
            viewer.last_range = frame.vrange
            if frame.colour is not None and not viewer.colour.decided:
                viewer.colours[viewer.var] = frame.colour
        self.query_one(PlotView).show(frame)
        if self._anim_timer is not None:
            self._shown.append(time.monotonic())
            if time.monotonic() - self._topbar_at > 0.5:
                self._update_topbar(self._topbar_extra)
        self.query_one("#header", Static).update(Text(frame.header))
        self.query_one("#stats", Static).update(Text(frame.stats))
        self.query_one("#status", Static).update(Text(frame.status))
        self.query_one(DimensionPanel).refresh()
        screen = self.screen
        if isinstance(screen, ColoursScreen):
            screen.refresh_range(frame.vrange)
        self._start_next()

    # ------------------------------------------------------------------ variables

    def schedule_select(self, name: str) -> None:
        if name == self.viewer.var:
            return
        if self._select_timer is not None:
            self._select_timer.stop()
        self._select_timer = self.set_timer(0.06, lambda: self.select_var(name))

    def select_var(self, name: str) -> None:
        if name == self.viewer.var or name not in self.src.vars:
            return
        self.viewer.select_var(name)
        if self._anim_timer is not None and self._anim_dim not in self.viewer.free_dims():
            self._stop_animation()
        self.view_changed()

    def action_search(self) -> None:
        search = self.query_one("#search", Input)
        search.add_class("visible")
        search.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search":
            self.query_one(VariableTree).rebuild(event.value, select=self.viewer.var)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search":
            tree = self.query_one(VariableTree)
            tree.focus()
            first = tree.first_variable()
            if first is not None and self.viewer.var not in tree._leaves:
                tree.highlight(first)

    def _close_search(self) -> bool:
        search = self.query_one("#search", Input)
        if not search.has_class("visible"):
            return False
        search.value = ""
        search.remove_class("visible")
        tree = self.query_one(VariableTree)
        tree.rebuild("", select=self.viewer.var)
        tree.focus()
        return True

    # ------------------------------------------------------------------ stepping & views

    def action_step_a(self, delta: int) -> None:
        a, _ = self.viewer.step_targets()
        if self.viewer.step(a, delta, wrap=True):
            self.view_changed()

    def action_step_b(self, delta: int) -> None:
        _, b = self.viewer.step_targets()
        if self.viewer.step(b, delta, wrap=True):
            self.view_changed()

    def action_cycle_view(self) -> None:
        if self.viewer.cycle_view():
            self.view_changed()
        else:
            self.notify("No other view for this variable. Press p to choose the axes.", timeout=3)

    def action_swap_axes(self) -> None:
        if self.viewer.swap_axes():
            self.view_changed()

    def put_on_axis(self, dim: str, axis: str) -> None:
        viewer = self.viewer
        try:
            if viewer.kind == "map":
                changed = viewer.assign_axis(dim, axis)
            elif viewer.kind == "line":
                if axis == "x":
                    viewer.set_axes(dim, None)
                elif dim != viewer.xdim:
                    viewer.set_axes(viewer.xdim, dim)
                changed = True
            else:
                changed = False
        except ValueError as exc:
            self.notify(str(exc), severity="warning")
            return
        if changed:
            self.view_changed()

    def action_back(self) -> None:
        if self._close_search():
            return
        if self.viewer.back():
            self.view_changed()

    # ------------------------------------------------------------------ zoom

    def action_zoom(self, factor: float) -> None:
        if self.viewer.zoom(factor):
            self.view_changed()

    def action_every_point(self) -> None:
        if self.viewer.every_point():
            self.view_changed()

    def action_fit(self) -> None:
        if self.viewer.fit():
            self.view_changed()

    def action_full_width(self) -> None:
        self.query_one("#sidebar").toggle_class("hidden")
        self.call_after_refresh(self.request_frame)

    def click_line(self, cx: int, cy: int) -> None:
        """Move the marker of a line plot to the clicked position."""
        f = self.last_frame
        if f is None or f.line_fig is None or f.request.kind != "line":
            return
        lf = f.line_fig
        d = self.src.dims[f.request.xdim]
        a, b = f.request.index_dict()[f.request.xdim]
        coord = d.values[a:b]
        lo, hi = float(min(coord)), float(max(coord))
        if lf.vertical:
            frac = 1 - (cy + 0.5) / lf.rows
            if not self.viewer.coord_up():
                frac = 1 - frac
        else:
            frac = (cx + 0.5) / lf.cols
        target = lo + frac * (hi - lo)
        i = a + int(abs(coord - target).argmin())
        if self.viewer.set_pos(f.request.xdim, i):
            self.view_changed()

    # ------------------------------------------------------------------ animation

    def action_play(self) -> None:
        if self._anim_timer is not None:
            self._stop_animation()
            return
        viewer = self.viewer
        dims = self.query_one(DimensionPanel)
        dim = None
        if dims.has_focus and dims.current in viewer.free_dims():
            dim = dims.current
        if dim is None:
            dim = viewer.step_targets()[0]
        if dim is None:
            self._cant_play("can't play: every dimension is plotted")
            return
        self._start_animation(dim)

    def _cant_play(self, message: str) -> None:
        """Say in the top bar, in red, why nothing is playing (for a few seconds)."""
        self._play_error = message
        self._update_topbar()

        def clear() -> None:
            if self._play_error == message:
                self._play_error = None
                self._update_topbar()

        self.set_timer(4, clear)

    def _shown_fps(self) -> float | None:
        """Frames per second that actually reached the screen over the last two seconds."""
        now = time.monotonic()
        recent = [t for t in self._shown if now - t <= 2.0]
        if len(recent) < 2 or now - self._anim_started < 1.5:
            return None
        return (len(recent) - 1) / (recent[-1] - recent[0]) if recent[-1] > recent[0] else None

    def _output_behind(self) -> bool:
        """Is the terminal still taking in earlier frames? (Then don't send another yet.)"""
        writer = getattr(getattr(self, "_driver", None), "_writer_thread", None)
        queue = getattr(writer, "_queue", None)
        return queue is not None and queue.qsize() > 0

    def _start_animation(self, dim: str) -> None:
        if self.src.dims[dim].size < 2:
            self._cant_play(f"can't play: only 1 {short_name(dim)} step")
            return
        if self._anim_timer is not None:
            self._anim_timer.stop()
        self._play_error = None
        self._anim_dim = dim
        self.viewer.anim_dim = dim
        self._anim_timer = self.set_interval(1 / FPS_STEPS[self._fps_i], self._anim_tick)
        self._shown.clear()
        self._anim_started = time.monotonic()
        self._update_topbar()
        self.request_frame()

    def _stop_animation(self) -> None:
        if self._anim_timer is not None:
            self._anim_timer.stop()
        self._anim_timer = None
        self._update_topbar()

    def _anim_tick(self) -> None:
        if self._building or self._output_behind():
            return  # skip a step rather than queue up frames the terminal hasn't drawn
        viewer = self.viewer
        dim = self._anim_dim
        if viewer.var is None or dim not in viewer.free_dims():
            self._stop_animation()
            return
        size = self.src.dims[dim].size
        viewer.set_pos(dim, (viewer.pos(dim) + 1) % size)
        self.view_changed()

    def action_speed(self, delta: int) -> None:
        self._fps_i = max(0, min(len(FPS_STEPS) - 1, self._fps_i + delta))
        if self._anim_timer is not None:
            self._anim_timer.stop()
            self._anim_timer = self.set_interval(1 / FPS_STEPS[self._fps_i], self._anim_tick)
            self._shown.clear()
            self._anim_started = time.monotonic()
        self._update_topbar()
        if self._anim_timer is None:
            self.notify(f"animation speed {FPS_STEPS[self._fps_i]} fps", timeout=2)

    # ------------------------------------------------------------------ colours & settings

    def action_colours(self) -> None:
        if self.viewer.var is not None:
            self.push_screen(ColoursScreen())

    def action_next_cmap(self) -> None:
        if self.viewer.var is not None:
            self.viewer.cycle_cmap(1)
            self.view_changed()

    def action_lock_range(self) -> None:
        if self.viewer.var is None:
            return
        mode = self.viewer.toggle_lock()
        self.notify("colour range locked" if mode == "locked" else "colour range follows each slice", timeout=2)
        self.view_changed()

    def action_plot_settings(self) -> None:
        if self.viewer.var is not None:
            self.push_screen(PlotSettingsScreen())

    def action_coastlines(self) -> None:
        viewer = self.viewer
        if not viewer.options.coastlines and not viewer.can_show_coastlines():
            self.notify("Coastlines need a longitude–latitude map.", timeout=3)
            return
        viewer.options.coastlines = not viewer.options.coastlines
        self.notify("coastlines on" if viewer.options.coastlines else "coastlines off", timeout=2)
        self.view_changed()

    def _start_anim_range_scan(self) -> None:
        """Scan every animation step of the current view for the "anim" colour range."""
        viewer = self.viewer
        key = viewer.anim_range_key()
        if key is None or key in viewer.anim_ranges or key == self._scan_key:
            return
        var, dim, rest, reduce = key
        self._scan_key = key
        self._scan_range(key, var, dict(rest), dim, reduce)

    @work(thread=True, group="scan", exclusive=True, exit_on_error=False)
    def _scan_range(self, key: tuple, var: str, index: dict, dim: str, reduce: tuple) -> None:
        worker = get_current_worker()
        label = f"scanning {var} over {short_name(dim)}"

        def progress(f: float) -> None:
            pct = int(f * 100) // 10 * 10
            if pct != self._scan_shown and not worker.is_cancelled:
                self._scan_shown = pct
                self.call_from_thread(self._update_topbar, f"{label} {pct}%")

        try:
            rng = self.src.value_range(var, index, dim, progress, lambda: worker.is_cancelled, reduce)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self.notify, f"Couldn't scan {var}: {exc}", severity="error")
            rng = None
        if not worker.is_cancelled:
            self.call_from_thread(self._scan_done, key, rng)

    def _scan_done(self, key: tuple, rng) -> None:
        if key == self._scan_key:
            self._scan_key = None
        self._scan_shown = -1
        self._update_topbar()
        if rng is None:
            return
        self.viewer.anim_ranges[key] = rng
        if key == self.viewer.anim_range_key():
            self.view_changed()

    # ------------------------------------------------------------------ popups

    def action_goto(self) -> None:
        if self.viewer.var is not None:
            self.push_screen(GoToScreen())

    def action_info(self) -> None:
        self.push_screen(InfoScreen(self.src, self.viewer.var))

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_save(self) -> None:
        viewer = self.viewer
        req = viewer.request()
        if req is None:
            return
        header = self.last_frame.header if self.last_frame else ""
        if req.kind == "map":
            idx = req.index_dict()
            nx = idx[req.xdim][1] - idx[req.xdim][0]
            ny = idx[req.ydim][1] - idx[req.ydim][0]
            note = f"Data: every point of the visible area ({nx} × {ny})."
        elif req.kind == "line":
            a, b = req.index_dict()[req.xdim]
            note = f"Data: every point of the line ({b - a} points)."
        else:
            note = ""
        note += " Uses your current colours, range, view and zoom. Name it .pdf or .svg for those formats."
        plot = SaveOptions(export.default_filename(self.src, req), export.default_title(self.src, req, header), note)
        anim = None
        dim = viewer.sweep_dim()
        if dim is not None and req.kind in ("map", "line"):
            d = self.src.dims[dim]
            fps = FPS_STEPS[self._fps_i]
            anim_note = f"{fps} fps (playback speed). {self._anim_range_note(req, dim)}"
            anim = AnimOptions(export.default_animation_filename(self.src, req, dim),
                               export.default_title(self.src, req, ""), anim_note, dim=dim,
                               first=d.label(0, with_lead=False), last=d.label(d.size - 1, with_lead=False),
                               fps=fps, frames=d.size)
        self.push_screen(SaveScreen(plot, anim, start_with_anim=self._anim_timer is not None))

    def _anim_range_note(self, req, dim: str) -> str:
        c = req.colour
        what = "Colour range" if req.kind == "map" else "Value axis"
        if c.mode == "auto":
            return f"{what}: auto, changes every frame (c → “min/max: fit animation” to fix it)."
        if c.mode == "anim":
            return f"{what}: all {short_name(dim)} steps, the same in every frame."
        return f"{what}: {c.mode}, the same in every frame."

    def save_png(self, path: str, title: str, size: tuple[float, float], dpi: int, screen: SaveScreen) -> None:
        req = self.viewer.request()
        vrange = self.last_frame.vrange if self.last_frame is not None else None
        self._export(req, path, title, size, dpi, vrange, screen)

    @staticmethod
    def _line_range(req) -> tuple[float, float] | None:
        c = req.colour
        if c.mode in ("locked", "manual"):
            return (c.vmin, c.vmax)
        return req.anim_range if c.mode == "anim" else None

    @work(thread=True, group="export", exit_on_error=False)
    def _export(self, req, path, title, size, dpi, vrange, screen) -> None:
        try:
            vr = self._line_range(req) if req.kind == "line" else vrange
            out = export.save_figure(self.src, req, path, title, size, dpi, vr)
        except Exception as exc:  # noqa: BLE001 - show any failure in the dialog
            self.call_from_thread(screen.finished, False, f"couldn't save: {exc}")
            return
        self.call_from_thread(screen.finished, True, out)
        self.call_from_thread(self.notify, f"Saved {out}", timeout=6)

    def save_animation(self, path: str, title: str, size: tuple[float, float], dpi: int, fps: float,
                       positions: list[int], screen: SaveScreen) -> None:
        viewer = self.viewer
        req = viewer.request()
        dim = viewer.sweep_dim()
        if req is None or dim is None:
            return
        key = viewer.anim_range_key() if req.colour.mode == "anim" else None
        self._save_worker = self._export_animation(req, dim, positions, key, path, title, size, dpi, fps, screen)

    def cancel_save(self) -> None:
        worker = getattr(self, "_save_worker", None)
        if worker is not None and worker.is_running:
            worker.cancel()
            self.notify("Saving cancelled.", timeout=3)

    @work(thread=True, group="export", exit_on_error=False)
    def _export_animation(self, req, dim, positions, key, path, title, size, dpi, fps, screen) -> None:
        import numpy as np

        from .frame import colour_range

        worker = get_current_worker()
        try:
            c = req.colour
            if c.mode == "auto":
                vrange = None
            elif req.kind == "line":
                vrange = self._line_range(req)
            else:
                rng = req.anim_range
                if c.mode == "anim" and rng is None and key is not None:
                    self.call_from_thread(screen.progress, f"scanning all {short_name(dim)} steps…")
                    rng = self.src.value_range(req.var, dict(key[2]), dim, reduce=key[3])
                vrange = colour_range(c, np.empty(0), rng)  # adds symmetric / log as on screen
            if req.kind == "line" and c.mode == "anim" and vrange is None and key is not None:
                vrange = self.src.value_range(req.var, dict(key[2]), dim, reduce=key[3])

            def progress(k: int, n: int) -> None:
                self.call_from_thread(screen.progress, f"saving frame {k} of {n}…")

            out = export.save_animation(self.src, req, dim, positions, path, title, size, dpi, fps, vrange,
                                        progress=progress, cancelled=lambda: worker.is_cancelled)
        except export.Cancelled:
            return
        except Exception as exc:  # noqa: BLE001 - show any failure in the dialog
            self.call_from_thread(screen.finished, False, f"couldn't save: {exc}")
            return
        self.call_from_thread(screen.finished, True, out)
        self.call_from_thread(self.notify, f"Saved {out}", timeout=6)

    # ------------------------------------------------------------------ action menus

    def _probe_title(self) -> str:
        viewer = self.viewer
        v = viewer.v
        where = ", ".join(f"{short_name(d)}={self.src.dims[d].label(viewer.pos(d), with_lead=False)}"
                          for d in viewer.plotted())
        value = ""
        if self.last_frame is not None and self.last_frame.probe_value is not None:
            from .data import format_number

            value = f" · {format_number(self.last_frame.probe_value, 6)} {v.units}".rstrip()
        return f"{v.name} at {where}{value}" if where else v.name

    def action_plot_menu(self) -> None:
        viewer = self.viewer
        if viewer.var is None:
            return
        options: list[tuple[str, str]] = []
        v = viewer.v
        if viewer.kind == "map":
            order = sorted(v.dims, key=lambda d: {Z: 0, TIME: 1}.get(viewer.role(d), 2))
            for d in order:
                role = viewer.role(d)
                if role == Z:
                    label = f"Vertical profile here ({d})"
                elif role == TIME:
                    label = f"Time series here ({d})"
                else:
                    label = f"Line along {d} here"
                options.append((f"line:{d}", label))
            options.append(("every", "Show every point around here (1)"))
            options.append(("zoom", "Zoom in here (+)"))
            plain = viewer.centring_axes(latitude=False)
            if plain:
                options.append(("centre", "Centre here: " + " + ".join(short_name(d) for d in plain)))
            both = viewer.centring_axes(latitude=True)
            if len(both) > len(plain):
                options.append(("centre_lat", "Centre here: " + " + ".join(short_name(d) for d in both)))
            if viewer.centred_axes():
                options.append(("uncentre", "Undo centring"))
        elif viewer.kind == "line":
            if viewer.stack:
                options.append(("back", "Back to the map (esc)"))
            options.append(("every", "Show every point around here (1)"))
        if viewer.is_zoomed():
            options.append(("fit", "Fit the whole field (0)"))
        options += [("colours", "Colours… (c)"), ("settings", "Plot settings… (p)"),
                    ("save", "Save plot or animation… (e)")]
        self.push_screen(ActionMenu(self._probe_title(), options), self._menu_choice)

    def action_variable_menu(self) -> None:
        tree = self.query_one(VariableTree)
        node = tree.cursor_node
        if node is None:
            return
        if node.data is None:
            node.toggle()
            return
        name = node.data
        if not self.src.vars[name].plottable:
            self.push_screen(InfoScreen(self.src, name))
            return
        self.select_var(name)
        options = [("plot", "Show the plot (tab)"), ("info", "Attributes… (i)"), ("colours", "Colours… (c)"),
                   ("settings", "Plot settings… (p)"), ("save", "Save plot or animation… (e)")]
        self.push_screen(ActionMenu(name, options), self._menu_choice)

    def action_dimension_menu(self) -> None:
        dims = self.query_one(DimensionPanel)
        d = dims.current
        if d is None:
            return
        viewer = self.viewer
        options: list[tuple[str, str]] = []
        if viewer.kind == "map":
            if d != viewer.xdim:
                options.append((f"x:{d}", "Put on the x axis (x)"))
            if d != viewer.ydim:
                options.append((f"y:{d}", "Put on the y axis (y)"))
            role = viewer.role(d)
            kind = "Vertical profile" if role == Z else "Time series" if role == TIME else "Line"
            options.append((f"line:{d}", f"{kind} along {d} at the probe"))
        if d in viewer.free_dims():
            options.append((f"anim:{d}", f"Animate along {d} (space)"))
            options.append((f"first:{d}", "Go to the first"))
            options.append((f"last:{d}", "Go to the last"))
        options.append(("goto", "Go to a value… (g)"))
        self.push_screen(ActionMenu(f"{d}", options), self._menu_choice)

    def _menu_choice(self, choice: str | None) -> None:
        if not choice:
            return
        viewer = self.viewer
        key, _, arg = choice.partition(":")
        if key == "line":
            if viewer.open_line(arg):
                self.query_one(PlotView).focus()
                self.view_changed()
        elif key == "every":
            self.action_every_point()
        elif key == "zoom":
            self.action_zoom(2)
        elif key == "fit":
            self.action_fit()
        elif key in ("centre", "centre_lat"):
            if viewer.centre_here(latitude=key == "centre_lat"):
                self.view_changed()
        elif key == "uncentre":
            if viewer.uncentre():
                self.view_changed()
        elif key == "back":
            self.action_back()
        elif key in ("x", "y"):
            self.put_on_axis(arg, key)
        elif key == "anim":
            self._start_animation(arg)
        elif key in ("first", "last"):
            size = self.src.dims[arg].size
            if viewer.set_pos(arg, 0 if key == "first" else size - 1):
                self.view_changed()
        elif key == "plot":
            self.query_one(PlotView).focus()
        elif key == "info":
            self.action_info()
        elif key == "colours":
            self.action_colours()
        elif key == "settings":
            self.action_plot_settings()
        elif key == "save":
            self.action_save()
        elif key == "goto":
            self.action_goto()
