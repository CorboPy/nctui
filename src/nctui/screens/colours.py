"""Colours popup: colour map, range mode, min/max, symmetric, log, clipping. Applies live."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rich.style import Style
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Input, Label, OptionList, RadioButton, RadioSet
from textual.widgets.option_list import Option

from .. import colormaps
from ..data import format_number
from ..frame import short_name
from ..state import RANGE_MODES

if TYPE_CHECKING:
    from ..app import NctuiApp

MODE_LABELS = {
    "auto": "auto: fit each frame",
    "locked": "locked: keep this range",
    "anim": "min/max: fit animation",
    "manual": "manual: type min / max",
}
MODE_HELP = {
    "auto": "Range of the frame on screen.",
    "locked": "The range on screen now.",
    "anim": "Min/max over all {n} {dim} steps.",
    "manual": "Applies on enter, tab or click-away.",
}


def mode_label(mode: str, dim: str | None) -> str:
    return MODE_LABELS[mode]

SIDE_PANEL_CSS = """
{name} {{
    align: left top;
    background: transparent;
}}
{name} > VerticalScroll {{
    width: 40;
    height: 100%;
    border: round $accent;
    border-title-style: bold;
    border-subtitle-color: $text-muted;
    background: $surface;
    padding: 0 1;
}}
{name} Label.section {{
    margin-top: 1;
    text-style: bold;
}}
{name} OptionList, {name} OptionList:focus {{
    height: auto;
    max-height: 18;
    border: none;
    padding: 0;
}}
{name} RadioSet {{
    border: none;
    padding: 0;
    width: 100%;
}}
{name} Checkbox {{
    border: none;
    padding: 0;
}}
{name} Select {{
    width: 100%;
}}
"""


def swatch(name: str, width: int = 20) -> Text:
    lut = colormaps.lut(name)
    text = Text()
    for i in np.linspace(0, 255, width).astype(int):
        r, g, b = (int(c) for c in lut[i])
        text.append("█", Style(color=f"rgb({r},{g},{b})"))
    return text


class ColoursScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "close"), Binding("c", "close", show=False)]
    DEFAULT_CSS = SIDE_PANEL_CSS.format(name="ColoursScreen") + """
    ColoursScreen #manual { height: 3; }
    ColoursScreen #manual.hidden { display: none; }
    ColoursScreen Input { width: 1fr; }
    ColoursScreen #mode-help { color: $text-muted; width: 100%; margin-bottom: 1; }
    """

    @property
    def nctui(self) -> NctuiApp:
        return self.app  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        viewer = self.nctui.viewer
        c = viewer.colour
        options = []
        for name in colormaps.NAMES:
            label = Text(name.ljust(10))
            label.append_text(swatch(name, 18))
            label.append(f" {colormaps.kind(name)[:3]}", style="dim")
            options.append(Option(label, id=name))
        lo, hi = viewer.last_range or (c.vmin, c.vmax)
        with VerticalScroll() as box:
            box.border_title = f"colours · {viewer.var}"
            box.border_subtitle = "changes apply live · esc done"
            yield Label("colour map", classes="section")
            yield OptionList(*options, id="cmaps")
            yield Label("range", classes="section")
            dim = viewer.sweep_dim()
            with RadioSet(id="mode"):
                for m in RANGE_MODES:
                    yield RadioButton(mode_label(m, dim), value=(m == c.mode))
            with Horizontal(id="manual", classes="" if c.mode == "manual" else "hidden"):
                yield Input(value=format_number(lo, 6) if lo is not None else "", placeholder="min", id="vmin")
                yield Input(value=format_number(hi, 6) if hi is not None else "", placeholder="max", id="vmax")
            yield Label(self._help(c.mode), id="mode-help")
            yield Checkbox("symmetric about zero", value=c.symmetric, id="symmetric")
            yield Checkbox("log scale", value=c.log, id="log")
            yield Checkbox("clip to 2nd–98th percentile", value=c.clip, id="clip")
            yield Label("", id="error")

    def on_mount(self) -> None:
        ol = self.query_one("#cmaps", OptionList)
        cmap = self.nctui.viewer.colour.cmap
        if cmap in colormaps.NAMES:
            ol.highlighted = colormaps.NAMES.index(cmap)
        ol.focus()

    def _error(self, text: str) -> None:
        self.query_one("#error", Label).update(Text(text, style="red"))

    def _help(self, mode: str) -> str:
        viewer = self.nctui.viewer
        dim = viewer.sweep_dim()
        if mode == "anim" and dim is None:
            return "Nothing to animate: same as auto."
        n = viewer.src.dims[dim].size if dim else 0
        return MODE_HELP[mode].format(n=n, dim=short_name(dim) if dim else "")

    def _show_mode(self, mode: str) -> None:
        self.query_one("#manual").set_class(mode != "manual", "hidden")
        self.query_one("#mode-help", Label).update(self._help(mode))

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        c = self.nctui.viewer.colour
        if event.option.id and event.option.id != c.cmap:
            c.cmap = event.option.id
            c.decided = True
            self.nctui.view_changed()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        event.stop()
        mode = RANGE_MODES[event.index]
        viewer = self.nctui.viewer
        c = viewer.colour
        if mode == c.mode:
            return
        self._show_mode(mode)
        if mode == "locked":
            if viewer.last_range is None:
                return
            c.mode = "locked"
            c.vmin, c.vmax = viewer.last_range
        elif mode == "manual":
            # start from the range on screen, then let the user edit it
            if viewer.last_range is not None:
                self.query_one("#vmin", Input).value = format_number(viewer.last_range[0], 6)
                self.query_one("#vmax", Input).value = format_number(viewer.last_range[1], 6)
            self._apply_manual()
            self.query_one("#vmin", Input).focus()
            return
        else:
            c.mode = mode  # auto, or anim (the app scans when it draws)
        self._error("")
        self.nctui.view_changed()

    def _apply_manual(self) -> None:
        try:
            lo = float(self.query_one("#vmin", Input).value)
            hi = float(self.query_one("#vmax", Input).value)
            self.nctui.viewer.set_manual_range(lo, hi)
        except ValueError as exc:
            self._error(str(exc) if "less" in str(exc) else "type numbers for min and max")
            return
        self._error("")
        self.nctui.view_changed()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._apply_manual()

    @on(Input.Blurred)
    def _input_left(self, event: Input.Blurred) -> None:
        """Tab, shift+tab or a click elsewhere applies a typed min/max."""
        event.stop()
        if self.nctui.viewer.colour.mode == "manual" and self.is_attached:
            self._apply_manual()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        c = self.nctui.viewer.colour
        setattr(c, event.checkbox.id, event.value)
        if event.checkbox.id == "symmetric":
            c.decided = True
        self.nctui.view_changed()

    def refresh_range(self, vrange: tuple[float, float] | None) -> None:
        """Called by the app after each frame so the min/max boxes show the current range."""
        if vrange is None:
            return
        vmin, vmax = self.query_one("#vmin", Input), self.query_one("#vmax", Input)
        if self.nctui.viewer.colour.mode != "manual" and not (vmin.has_focus or vmax.has_focus):
            vmin.value = format_number(vrange[0], 6)
            vmax.value = format_number(vrange[1], 6)

    def action_close(self) -> None:
        self.dismiss(None)
