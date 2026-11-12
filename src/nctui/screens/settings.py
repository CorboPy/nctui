"""Plot settings popup: axes, aspect ratio, flips, how points are combined, coastlines. Live."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Input, Label, RadioButton, RadioSet, Select

from ..data import format_number
from ..frame import short_name
from ..reduce import METHOD_LABELS, METHODS
from ..state import ASPECTS
from .colours import SIDE_PANEL_CSS

if TYPE_CHECKING:
    from ..app import NctuiApp

LINE = "__line__"
ASPECT_LABELS = {
    "auto": "auto (equal when units match)",
    "equal": "equal (or square cells)",
    "fill": "fill the pane",
}
ASPECT_NOTE = "equal, if the units differ: square cells."
METHOD_HELP = {
    "mean": "mean (area average)",
    "max": "max (keeps peaks, thin layers)",
    "min": "min (keeps troughs)",
    "maxabs": "max |value| (either sign)",
    "nearest": "nearest (real values)",
}


class PlotSettingsScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "close"), Binding("p", "close", show=False)]
    DEFAULT_CSS = SIDE_PANEL_CSS.format(name="PlotSettingsScreen") + """
    PlotSettingsScreen #centre { height: auto; }
    PlotSettingsScreen #centre.hidden { display: none; }
    PlotSettingsScreen #centre Horizontal { height: 3; }
    PlotSettingsScreen #centre Horizontal Label { width: 15; height: 3; content-align: left middle; }
    PlotSettingsScreen #centre Input { width: 1fr; }
    PlotSettingsScreen .note { color: $text-muted; width: 100%; }
    """

    @property
    def nctui(self) -> NctuiApp:
        return self.app  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        viewer = self.nctui.viewer
        dims = viewer.live_dims()  # a reduced dimension can't go on an axis
        opts = viewer.options
        with VerticalScroll() as box:
            box.border_title = "plot settings"
            box.border_subtitle = "changes apply live · esc done"
            if len(dims) >= 1:
                yield Label("x axis", classes="section")
                yield Select([(d, d) for d in dims], value=viewer.xdim if viewer.xdim in dims else dims[0],
                             allow_blank=False, id="xdim")
                yield Label("y axis", classes="section")
                yopts = [(d, d) for d in dims] + [("none: line plot", LINE)]
                yval = viewer.ydim if viewer.kind == "map" and viewer.ydim in dims else LINE
                yield Select(yopts, value=yval, allow_blank=False, id="ydim")
            yield Label("aspect ratio", classes="section")
            with RadioSet(id="aspect"):
                for a in ASPECTS:
                    yield RadioButton(ASPECT_LABELS[a], value=(a == opts.aspect))
            yield Label(ASPECT_NOTE, classes="note")
            yield Checkbox("flip x axis", value=opts.flip_x, id="flip_x")
            yield Checkbox("flip y axis", value=opts.flip_y, id="flip_y")
            with Vertical(id="centre"):
                yield Label("centre on", classes="section")
                for axis in ("x", "y"):
                    with Horizontal():
                        yield Label(axis, id=f"centre-{axis}-label")
                        yield Input(id=f"centre-{axis}")
                yield Label("Value for the middle; blank: as in the file.", classes="note")
            yield Label("when points share a pixel", classes="section")
            with RadioSet(id="method"):
                for m in METHODS:
                    yield RadioButton(METHOD_HELP.get(m, METHOD_LABELS[m]), value=(m == opts.method))
            yield Label("overlays", classes="section")
            cb = Checkbox("coastlines (lat/lon maps)", value=opts.coastlines, id="coastlines")
            yield cb
            yield Label("", id="error")

    def on_mount(self) -> None:
        self._refresh_centre()
        sel = self.query("Select")
        (sel.first() if sel else self.query_one(RadioSet)).focus()

    def _axes(self) -> dict[str, str | None]:
        viewer = self.nctui.viewer
        return {"x": viewer.xdim, "y": viewer.ydim} if viewer.kind == "map" else {}

    def _refresh_centre(self) -> None:
        """Show which axes can be centred, and where they are centred now."""
        viewer = self.nctui.viewer
        axes = self._axes()
        self.query_one("#centre").set_class(not axes, "hidden")
        for axis, dim in axes.items():
            box = self.query_one(f"#centre-{axis}", Input)
            self.query_one(f"#centre-{axis}-label", Label).update(f"{axis} {short_name(dim)}"[:15])
            can = viewer.can_centre(dim)
            box.disabled = not can
            centre = viewer.src.centre(dim) if dim else None
            box.value = format_number(centre, 8) if centre is not None else ""
            box.placeholder = "as in file" if can else "can't wrap"

    def _apply_centre(self, box: Input) -> None:
        axis = box.id.removeprefix("centre-") if box.id else ""
        dim = self._axes().get(axis)
        if dim is None or box.disabled:
            return
        viewer = self.nctui.viewer
        text = box.value.strip()
        try:
            value = float(text) if text else None
        except ValueError:
            self._error(f"[red]type a number for the {axis} centre, or leave it blank[/]")
            return
        if value == viewer.src.centre(dim):
            return
        try:
            viewer.centre_on(dim, value)
        except ValueError as exc:
            self._error(f"[red]{exc}[/]")
            return
        self._error("")
        self.nctui.view_changed()

    @on(Input.Submitted)
    @on(Input.Blurred)
    def _centre_typed(self, event: Input.Submitted | Input.Blurred) -> None:
        event.stop()
        if self.is_attached:
            self._apply_centre(event.input)

    def _error(self, text: str) -> None:
        self.query_one("#error", Label).update(text)

    def on_select_changed(self, event: Select.Changed) -> None:
        event.stop()
        viewer = self.nctui.viewer
        x = self.query_one("#xdim", Select).value
        y = self.query_one("#ydim", Select).value
        y = None if y == LINE else y
        if (x, y) == (viewer.xdim, viewer.ydim if viewer.kind == "map" else None):
            return
        try:
            viewer.set_axes(str(x), None if y is None else str(y))
        except ValueError as exc:
            self._error(f"[red]{exc}[/]")
            return
        self._error("")
        self._refresh_centre()
        self.nctui.view_changed()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        event.stop()
        opts = self.nctui.viewer.options
        if event.radio_set.id == "aspect":
            opts.aspect = ASPECTS[event.index]
        elif event.radio_set.id == "method":
            opts.method = METHODS[event.index]
        self.nctui.view_changed()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        setattr(self.nctui.viewer.options, event.checkbox.id, event.value)
        self.nctui.view_changed()

    def action_close(self) -> None:
        self.dismiss(None)
