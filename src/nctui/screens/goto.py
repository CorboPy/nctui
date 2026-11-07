"""Go to popup: type a value (or a range, for plotted dimensions) for any dimension."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label

if TYPE_CHECKING:
    from ..app import NctuiApp


def parse_range(dim, text: str) -> tuple[int, int] | None:
    """``"all"``/blank → whole axis; ``"a:b"`` → window; a single value → None (probe only)."""
    text = text.strip()
    if text in ("", "all", "*", ":"):
        return 0, dim.size
    if ":" in text:
        a, b = text.split(":", 1)
        ia = dim.find(a) if a.strip() else 0
        ib = dim.find(b) if b.strip() else dim.size - 1
        lo, hi = min(ia, ib), max(ia, ib)
        return lo, hi + 1
    return None


class GoToScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "close")]
    DEFAULT_CSS = """
    GoToScreen {
        align: center middle;
        background: $background 40%;
    }
    GoToScreen > Vertical {
        width: 76;
        height: auto;
        border: round $accent;
        border-title-style: bold;
        border-subtitle-color: $text-muted;
        background: $surface;
        padding: 0 1;
    }
    GoToScreen Grid {
        grid-size: 2;
        grid-columns: 20 1fr;
        grid-rows: 3;
        height: auto;
    }
    GoToScreen Grid Label {
        height: 3;
        content-align: left middle;
    }
    GoToScreen .hint { color: $text-muted; }
    """

    @property
    def nctui(self) -> NctuiApp:
        return self.app  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        viewer = self.nctui.viewer
        src = viewer.src
        plotted = viewer.plotted()
        with Vertical() as box:
            box.border_title = "go to"
            box.border_subtitle = "enter applies · esc cancels"
            yield Label("Type a value, a date, #index, or for plotted axes a range like 0:2000.",
                        classes="hint")
            with Grid():
                for d in viewer.v.dims:
                    dim = src.dims[d]
                    if d in plotted:
                        a, b = viewer.window(d)
                        value = "all" if (a, b) == (0, dim.size) else (
                            f"{dim.label(a, with_lead=False)}:{dim.label(b - 1, with_lead=False)}")
                        tag = " (range)"
                    else:
                        value = dim.label(viewer.pos(d), with_lead=False)
                        tag = ""
                    yield Label(f"{d}{tag}"[:20])
                    yield Input(value=value, id=f"dim-{list(viewer.v.dims).index(d)}")
            yield Label("", id="error")

    def on_mount(self) -> None:
        inputs = self.query(Input)
        if inputs:
            inputs.first().focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        viewer = self.nctui.viewer
        src = viewer.src
        plotted = viewer.plotted()
        changes: list[tuple[str, str, object]] = []
        try:
            for i, d in enumerate(viewer.v.dims):
                text = self.query_one(f"#dim-{i}", Input).value
                dim = src.dims[d]
                if d in plotted:
                    rng = parse_range(dim, text)
                    if rng is None:
                        changes.append(("pos", d, dim.find(text)))
                    else:
                        changes.append(("win", d, rng))
                else:
                    changes.append(("pos", d, dim.find(text)))
        except (ValueError, IndexError) as exc:
            self.query_one("#error", Label).update(f"[red]{d}: {exc}[/]")
            return
        for kind, d, val in changes:
            if kind == "pos":
                viewer.set_pos(d, int(val))
            else:
                a, b = val
                viewer.set_window(d, a, b)
                p = viewer.pos(d)
                if not (a <= p < b):
                    viewer.set_pos(d, (a + b - 1) // 2)
        self.nctui.view_changed()
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)

