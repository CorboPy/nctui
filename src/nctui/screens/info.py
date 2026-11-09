"""Read-only popups: variable/file information and help."""

from __future__ import annotations

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from ..data import Source

_CSS = """
{name} {{
    align: center middle;
    background: $background 40%;
}}
{name} > VerticalScroll {{
    width: 90%;
    max-width: 110;
    height: 85%;
    border: round $accent;
    border-title-style: bold;
    border-subtitle-color: $text-muted;
    background: $surface;
    padding: 0 1;
}}
"""


def _attr_table(title: str, attrs: dict) -> Table:
    t = Table(title=title, title_justify="left", title_style="bold", show_header=False,
              box=None, padding=(0, 1), expand=True)
    t.add_column(style="cyan", no_wrap=True, ratio=1)
    t.add_column(ratio=3)
    for k, v in attrs.items():
        t.add_row(str(k), str(v))
    if not attrs:
        t.add_row("", Text("(none)", style="dim"))
    return t


class InfoScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "close"), Binding("i", "close", show=False),
                Binding("q", "close", show=False)]
    DEFAULT_CSS = _CSS.format(name="InfoScreen")

    def __init__(self, src: Source, var: str | None):
        super().__init__()
        self.src = src
        self.var = var

    def compose(self) -> ComposeResult:
        parts = []
        if self.var is not None:
            v = self.src.vars[self.var]
            dims = Table(title="dimensions", title_justify="left", title_style="bold", box=None,
                         padding=(0, 1), expand=True)
            for col in ("name", "size", "kind", "range", "units"):
                dims.add_column(col, style="cyan" if col == "name" else None)
            for d in v.dims:
                dim = self.src.dims[d]
                rng = ""
                if dim.size:
                    rng = f"{dim.label(0, with_lead=False)} … {dim.label(dim.size - 1, with_lead=False)}"
                dims.add_row(d, str(dim.size), dim.role, rng, dim.units)
            parts += [
                Text(f"{v.name}  {v.dtype}{list(v.shape)}", style="bold"),
                Text(""),
                _attr_table("attributes", v.attrs),
                Text(""),
                dims,
                Text(""),
            ]
        files = Table(title="files", title_justify="left", title_style="bold", show_header=False,
                      box=None, padding=(0, 1))
        for p in self.src.paths:
            files.add_row(p)
        parts += [files, Text(""), _attr_table("global attributes", self.src.file_attrs())]
        for w in self.src.warnings:
            parts.append(Text(f"⚠ {w}", style="yellow"))
        with VerticalScroll() as box:
            box.border_title = f"info · {self.var}" if self.var else "info"
            box.border_subtitle = "esc closes"
            yield Static(Group(*parts))

    def on_mount(self) -> None:
        self.query_one(VerticalScroll).focus()

    def action_close(self) -> None:
        self.dismiss(None)


HELP = [
    ("Getting around", [
        ("tab / shift+tab", "move between Variables, Dimensions and Plot (or click)"),
        ("↑ ↓", "Variables: choose a variable (it plots as you go)"),
        ("/", "search variables by name, long_name or standard_name"),
        ("enter", "actions for whatever is selected"),
        ("esc", "close a popup, or go back from a line plot to the map"),
    ]),
    ("Slicing", [
        ("[ ]", "step time back / forward (wraps round)"),
        ("{ }", "step the vertical level down / up (wraps round)"),
        ("← →", "Dimensions: step the highlighted dimension, wrapping round (shift = ×10, home/end)"),
        ("x  y", "Dimensions: put the highlighted dimension on the x or y axis"),
        ("v", "cycle views: map → section (z–x) → section (z–y)"),
        ("t", "swap the plot axes"),
        ("g", "go to a value: type a date, a height, #index, or a range like 0:2000"),
        ("space", "play / pause animation (along time, or the highlighted dimension)"),
        ("< >", "slower / faster animation"),
    ]),
    ("Probe and zoom", [
        ("← → ↑ ↓", "Plot: move the probe (shift = ×10); moving past the edge pans"),
        ("click", "put the probe there; drag a box to zoom into it"),
        ("enter", "Plot: profile / time series / line through the probe; centre the map here"),
        ("+ -", "zoom in / out around the probe"),
        ("1", "show every point (one data point per pixel) around the probe"),
        ("0", "fit the whole field"),
        ("f", "full-width plot (hide the left panes)"),
    ]),
    ("Colours and plot", [
        ("c", "Colours popup: colour map, range (e.g. one range for a whole animation), log"),
        ("m", "next colour map"),
        ("r", "lock / unlock the colour range (handy when animating)"),
        ("p", "Plot settings popup: axes, aspect, centre (wrap longitude / latitude), coastlines"),
        ("o", "coastlines on / off (lat/lon maps)"),
    ]),
    ("Other", [
        ("e", "save the view as PNG/PDF/SVG, or an animation as MP4/GIF (every data point)"),
        ("i", "attributes of the variable and file"),
        ("?", "this help"),
        ("q", "quit"),
    ]),
]

RESOLUTION_NOTE = (
    "The line under the plot shows how the data was fitted to the screen: "
    "\"z 301→104 ÷2.9 (mean)\" means about 3 levels share each pixel row, averaged; "
    "\"1:1\" means every point is shown. Change how points are combined in Plot settings (p), "
    "press 1 to see every point, or make the terminal font smaller for more pixels."
)


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "close"), Binding("question_mark", "close", show=False),
                Binding("q", "close", show=False)]
    DEFAULT_CSS = _CSS.format(name="HelpScreen")

    def compose(self) -> ComposeResult:
        t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        t.add_column(style="bold cyan", no_wrap=True, width=16)
        t.add_column(ratio=1)
        for i, (section, rows) in enumerate(HELP):
            if i:
                t.add_row("", "")
            t.add_row(Text(section, style="bold"), "")
            for k, d in rows:
                t.add_row(k, d)
        parts = [t, Text(""), Text(RESOLUTION_NOTE, style="dim")]
        with VerticalScroll() as box:
            box.border_title = "nctui keys"
            box.border_subtitle = "esc closes"
            yield Static(Group(*parts))

    def on_mount(self) -> None:
        self.query_one(VerticalScroll).focus()

    def action_close(self) -> None:
        self.dismiss(None)
