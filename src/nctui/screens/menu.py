"""A small Cylc-style action menu for whatever is selected."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option


class ActionMenu(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "close", "close"), Binding("q", "close", show=False)]

    DEFAULT_CSS = """
    ActionMenu {
        align: center middle;
        background: $background 30%;
    }
    ActionMenu > Vertical {
        width: auto;
        min-width: 44;
        max-width: 80;
        height: auto;
        border: round $accent;
        border-title-color: $text;
        border-title-style: bold;
        background: $surface;
        padding: 0 1;
    }
    ActionMenu OptionList, ActionMenu OptionList:focus {
        height: auto;
        max-height: 24;
        border: none;
        padding: 0;
        background: $surface;
    }
    ActionMenu .hint {
        width: 100%;
        color: $text-muted;
        text-align: right;
    }
    """

    def __init__(self, title: str, options: list[tuple[str, str]]):
        super().__init__()
        self.menu_title = title
        self.options = options

    def compose(self) -> ComposeResult:
        with Vertical() as box:
            box.border_title = self.menu_title
            yield OptionList(*[Option(label, id=key) for key, label in self.options])
            yield Label("enter chooses · esc closes", classes="hint")

    def on_mount(self) -> None:
        ol = self.query_one(OptionList)
        ol.highlighted = 0
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id)

    def action_close(self) -> None:
        self.dismiss(None)
