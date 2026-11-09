"""The dimensions pane: one entry per dimension of the current variable."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.widget import Widget

from ..data import format_number

if TYPE_CHECKING:
    from ..app import NctuiApp


class DimensionPanel(Widget, can_focus=True):
    """Shows where you are along each dimension; ←→ steps the highlighted one."""

    BINDINGS = [
        Binding("up", "cursor(-1)", "dimension", show=False),
        Binding("down", "cursor(1)", show=False),
        Binding("left", "step(-1)", "step", key_display="←→"),
        Binding("right", "step(1)", show=False),
        Binding("shift+left", "step(-10)", show=False),
        Binding("shift+right", "step(10)", show=False),
        Binding("home", "jump(0)", show=False),
        Binding("end", "jump(-1)", show=False),
        Binding("x", "axis('x')", "x axis"),
        Binding("y", "axis('y')", "y axis"),
        Binding("enter", "app.dimension_menu", "actions", key_display="enter"),
    ]

    DEFAULT_CSS = """
    DimensionPanel {
        height: auto;
        min-height: 3;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursor = 0

    @property
    def nctui(self) -> NctuiApp:
        return self.app  # type: ignore[return-value]

    def dims(self) -> list[str]:
        v = self.nctui.viewer
        return list(v.v.dims) if v.var is not None else []

    @property
    def current(self) -> str | None:
        dims = self.dims()
        if not dims:
            return None
        self.cursor = min(self.cursor, len(dims) - 1)
        return dims[self.cursor]

    def get_content_height(self, container, viewport, width) -> int:
        return max(1, 2 * len(self.dims()))

    def render(self) -> Text:
        viewer = self.nctui.viewer
        if viewer.var is None:
            return Text("no variable", style="dim")
        src = viewer.src
        width = max(10, self.size.width)
        plotted = viewer.plotted()
        out = Text(no_wrap=True, overflow="ellipsis")
        for i, d in enumerate(self.dims()):
            dim = src.dims[d]
            selected = i == self.cursor and self.has_focus
            name_style = "bold reverse" if selected else "bold"
            line1 = Text(no_wrap=True)
            line1.append(d[: width - 12], style=name_style)
            if d in plotted:
                if viewer.kind == "map":
                    axis = "x" if d == viewer.xdim else "y"
                    tag = f"plot {axis}"
                else:
                    tag = "line"
                pad = max(1, width - len(line1.plain) - len(tag))
                line1.append(" " * pad)
                line1.append(tag, style="bold magenta")
                a, b = viewer.window(d)
                span = "all" if (a, b) == (0, dim.size) else f"{a}–{b - 1} of {dim.size}"
                where = f"{span} · probe {dim.label(viewer.pos(d), with_lead=False)}"
                centre = src.centre(d)
                if centre is not None:
                    where += f" · centred {format_number(centre)}"
                line2 = Text("  " + where, style="dim")
            else:
                i_pos = viewer.pos(d)
                counter = f"◀ {i_pos + 1:>{len(str(dim.size))}}/{dim.size} ▶"
                pad = max(1, width - len(line1.plain) - len(counter))
                line1.append(" " * pad)
                line1.append(counter, style="cyan")
                line2 = Text("  " + dim.label(i_pos), style="")
            line1.truncate(width)
            line2.truncate(width, overflow="ellipsis")
            out.append(line1)
            out.append("\n")
            out.append(line2)
            if i < len(self.dims()) - 1:
                out.append("\n")
        return out

    # ------------------------------------------------------------------ actions

    def action_cursor(self, delta: int) -> None:
        dims = self.dims()
        if dims:
            self.cursor = (self.cursor + delta) % len(dims)
            self.refresh()

    def action_step(self, delta: int) -> None:
        d = self.current
        if d is None:
            return
        viewer = self.nctui.viewer
        if d in viewer.plotted():
            # stepping a plotted dimension moves the probe along it
            if viewer.step(d, delta * viewer.probe_step(d), wrap=True):
                self.nctui.view_changed()
            return
        if viewer.step(d, delta, wrap=True):  # ◀ on the first step goes to the last
            self.nctui.view_changed()

    def action_jump(self, where: int) -> None:
        d = self.current
        if d is None:
            return
        viewer = self.nctui.viewer
        size = viewer.src.dims[d].size
        if viewer.set_pos(d, 0 if where == 0 else size - 1):
            self.nctui.view_changed()

    def action_axis(self, axis: str) -> None:
        d = self.current
        if d is not None:
            self.nctui.put_on_axis(d, axis)

    def on_focus(self) -> None:
        self.refresh()

    def on_blur(self) -> None:
        self.refresh()

    def on_click(self, event: events.Click) -> None:
        offset = event.get_content_offset(self)
        if offset is None:
            return
        row = offset.y // 2
        dims = self.dims()
        if not (0 <= row < len(dims)):
            return
        self.cursor = row
        d = dims[row]
        viewer = self.nctui.viewer
        if offset.y % 2 == 0 and d not in viewer.plotted():
            width = self.content_size.width
            counter_len = len(f"◀ {viewer.pos(d) + 1:>{len(str(viewer.src.dims[d].size))}}"
                              f"/{viewer.src.dims[d].size} ▶")
            x = offset.x
            start = width - counter_len
            if start <= x < start + 2:
                self.action_step(-1)
                return
            if width - 2 <= x < width:
                self.action_step(1)
                return
        self.refresh()
