"""Save popup: this plot (PNG/PDF/SVG) or an animation of it (MP4/GIF), pre-filled from the view."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet

from ..export import VIDEO_EXTENSIONS, can_write_video

if TYPE_CHECKING:
    from ..app import NctuiApp

PLOT_EXTENSIONS = (".png", ".pdf", ".svg")
ANIM_FORMATS = (".mp4", ".gif")
PLOT_DPI, ANIM_DPI = "150", "100"


@dataclass
class SaveOptions:
    filename: str
    title: str
    note: str


@dataclass
class AnimOptions(SaveOptions):
    dim: str
    first: str  # labels of the first and last positions along dim
    last: str
    fps: float
    frames: int


class SaveScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "cancel")]
    DEFAULT_CSS = """
    SaveScreen {
        align: center middle;
        background: $background 40%;
    }
    SaveScreen > Vertical {
        width: 84;
        height: auto;
        border: round $accent;
        border-title-style: bold;
        border-subtitle-color: $text-muted;
        background: $surface;
        padding: 0 1;
    }
    SaveScreen Grid {
        grid-size: 2;
        grid-columns: 8 1fr;
        grid-rows: 3;
        height: auto;
    }
    SaveScreen Grid > Label {
        height: 3;
        content-align: left middle;
    }
    SaveScreen RadioSet, SaveScreen RadioSet:focus {
        layout: horizontal;
        border: none;  /* the default focus border is 2 rows tall and would hide the buttons */
        height: 1;
        margin-top: 1;
        width: 100%;
    }
    SaveScreen RadioSet > RadioButton, SaveScreen RadioSet:focus > RadioButton { width: auto; margin-right: 3; }
    SaveScreen .row { height: 3; }
    SaveScreen .row Input { width: 10; }
    SaveScreen .row Input.wide { width: 22; }
    SaveScreen .row Label { height: 3; content-align: left middle; padding: 0 1; }
    SaveScreen .hidden { display: none; }
    SaveScreen .note { color: $text-muted; width: 100%; }
    SaveScreen #buttons { height: 3; align-horizontal: right; }
    SaveScreen Button { margin-left: 1; }
    """

    def __init__(self, plot: SaveOptions, anim: AnimOptions | None, start_with_anim: bool = False):
        super().__init__()
        self.plot = plot
        self.anim_opts = anim
        self.anim = start_with_anim and anim is not None
        self.saving = False

    @property
    def nctui(self) -> NctuiApp:
        return self.app  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        a = self.anim_opts
        opts = a if self.anim and a is not None else self.plot
        with Vertical() as box:
            box.border_subtitle = "enter saves · esc cancels"
            with Grid():
                yield Label("save")
                with RadioSet(id="what"):
                    yield RadioButton("this plot", value=not self.anim)
                    yield RadioButton(f"animation along {a.dim} ({a.frames} frames)" if a else
                                      "animation (nothing to animate)", value=self.anim, disabled=a is None)
                yield Label("format", id="format-label")
                video = can_write_video()
                gif = os.path.splitext(opts.filename)[1].lower() == ".gif" or not video
                with RadioSet(id="format"):
                    yield RadioButton("mp4" if video else "mp4 (needs ffmpeg)", value=not gif, disabled=not video)
                    yield RadioButton("gif", value=gif)
                yield Label("file")
                yield Input(opts.filename, id="file")
                yield Label("title")
                yield Input(opts.title.replace("\n", " | "), id="title")
                yield Label("size")
                with Horizontal(classes="row"):
                    yield Input("8", id="width", type="number")
                    yield Label("×")
                    yield Input("6", id="height", type="number")
                    yield Label("in   dpi")
                    yield Input(ANIM_DPI if self.anim else PLOT_DPI, id="dpi", type="integer")
                yield Label("frames", id="frames-label")
                with Horizontal(classes="row", id="frames"):
                    yield Input(a.first if a else "", id="first", classes="wide")
                    yield Label("to")
                    yield Input(a.last if a else "", id="last", classes="wide")
                    yield Label("fps")
                    yield Input(f"{a.fps:g}" if a else "5", id="fps", type="number")
            yield Label(opts.note, id="note", classes="note")
            yield Label("", id="message")
            with Horizontal(id="buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self._show_mode()
        self.query_one("#file", Input).focus()

    def _show_mode(self) -> None:
        anim = self.anim
        self.query_one(Vertical).border_title = "save animation" if anim else "save plot"
        for wid in ("#frames-label", "#frames", "#format-label", "#format"):
            self.query_one(wid).set_class(not anim, "hidden")
        opts = self.anim_opts if anim and self.anim_opts else self.plot
        self.query_one("#note", Label).update(opts.note)

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        event.stop()
        if event.radio_set.id == "format":
            self._set_extension(ANIM_FORMATS[event.index])
            return
        anim = event.index == 1 and self.anim_opts is not None
        if anim == self.anim:
            return
        self.anim = anim
        new, old = (self.anim_opts, self.plot) if anim else (self.plot, self.anim_opts)
        assert new is not None and old is not None
        file = self.query_one("#file", Input)
        if file.value == old.filename:
            file.value = new.filename
        else:  # keep the user's name, change the kind of file
            root, ext = os.path.splitext(file.value)
            if anim and ext.lower() in PLOT_EXTENSIONS:
                file.value = root + os.path.splitext(new.filename)[1]
            elif not anim and ext.lower() in (*VIDEO_EXTENSIONS, ".gif"):
                file.value = root + ".png"
        if anim:
            self._set_extension(self._chosen_format())
        title = self.query_one("#title", Input)
        if title.value == old.title.replace("\n", " | "):
            title.value = new.title.replace("\n", " | ")
        dpi = self.query_one("#dpi", Input)
        if dpi.value == (PLOT_DPI if anim else ANIM_DPI):
            dpi.value = ANIM_DPI if anim else PLOT_DPI
        self._show_mode()

    def _chosen_format(self) -> str:
        return ANIM_FORMATS[self.query_one("#format", RadioSet).pressed_index]

    def _set_extension(self, ext: str) -> None:
        """Give the file name the chosen animation format's extension."""
        file = self.query_one("#file", Input)
        root, old = os.path.splitext(file.value)
        if old.lower() != ext:
            file.value = (root if old.lower() in (*VIDEO_EXTENSIONS, ".gif", *PLOT_EXTENSIONS) else file.value) + ext

    def on_input_changed(self, event: Input.Changed) -> None:
        """Typing a .gif or .mp4 name picks that format."""
        event.stop()
        if event.input.id != "file" or not self.anim:
            return
        ext = os.path.splitext(event.value)[1].lower()
        if ext in ANIM_FORMATS:
            button = self.query_one("#format", RadioSet).query(RadioButton)[ANIM_FORMATS.index(ext)]
            if not button.value and not button.disabled:
                button.value = True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._save()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "save":
            self._save()
        else:
            self.action_close()

    def _save(self) -> None:
        if self.saving:
            return
        msg = self.query_one("#message", Label)
        try:
            path = self.query_one("#file", Input).value.strip()
            if not path:
                raise ValueError("type a file name")
            width = float(self.query_one("#width", Input).value)
            height = float(self.query_one("#height", Input).value)
            dpi = int(self.query_one("#dpi", Input).value)
            if not (0 < width <= 100 and 0 < height <= 100 and 10 <= dpi <= 2000):
                raise ValueError("size must be 0–100 in and dpi 10–2000")
            positions: list[int] = []
            fps = 0.0
            if self.anim and self.anim_opts is not None:
                dim = self.nctui.src.dims[self.anim_opts.dim]
                a = dim.find(self.query_one("#first", Input).value)
                b = dim.find(self.query_one("#last", Input).value)
                positions = list(range(a, b + 1)) if b >= a else list(range(a, b - 1, -1))
                fps = float(self.query_one("#fps", Input).value)
                if not 0 < fps <= 60:
                    raise ValueError("fps must be more than 0 and at most 60")
        except ValueError as exc:
            msg.update(f"[red]{exc}[/]")
            return
        title = self.query_one("#title", Input).value.replace(" | ", "\n")
        self.saving = True
        self.query_one("#save", Button).disabled = True
        msg.update("saving…")
        if self.anim:
            self.nctui.save_animation(path, title, (width, height), dpi, fps, positions, self)
        else:
            self.nctui.save_png(path, title, (width, height), dpi, self)

    def progress(self, text: str) -> None:
        if self.is_attached:
            self.query_one("#message", Label).update(text)

    def finished(self, ok: bool, text: str) -> None:
        if not self.is_attached:
            return
        self.saving = False
        if ok:
            self.dismiss(None)
        else:
            self.query_one("#save", Button).disabled = False
            self.query_one("#message", Label).update(f"[red]{text}[/]")

    def action_close(self) -> None:
        if self.saving:
            self.nctui.cancel_save()
        self.dismiss(None)
