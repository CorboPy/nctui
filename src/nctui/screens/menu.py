"""A small Cylc-style action menu for whatever is selected."""

from __future__ import annotations

from dataclasses import dataclass

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option


@dataclass(frozen=True)
class Entry:
    """One choice: ``id`` is what the menu returns, ``key`` the shortcut shown beside it."""

    id: str
    label: str
    key: str = ""
    enabled: bool = True


Section = tuple[str, list[Entry]]


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
        max-height: 30;
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

    def __init__(self, title: str, sections: list[Section]):
        super().__init__()
        self.menu_title = title
        self.sections = [(name, entries) for name, entries in sections if entries]

    @property
    def entries(self) -> dict[str, Entry]:
        return {e.id: e for _, entries in self.sections for e in entries}

    def _options(self) -> list[Option | None]:
        width = max((cell_len(e.label) for _, es in self.sections for e in es), default=0)
        out: list[Option | None] = []
        for i, (name, entries) in enumerate(self.sections):
            if i:
                out.append(None)
            if name:
                out.append(Option(Text(name, style="bold dim"), disabled=True))
            for e in entries:
                prompt = Text(e.label + " " * (width - cell_len(e.label)))
                if e.key:
                    prompt.append("  " + e.key, style="dim")
                out.append(Option(prompt, id=e.id, disabled=not e.enabled))
        return out

    def compose(self) -> ComposeResult:
        with Vertical() as box:
            box.border_title = self.menu_title
            yield OptionList(*self._options())
            yield Label("enter chooses · esc closes", classes="hint")

    def on_mount(self) -> None:
        ol = self.query_one(OptionList)
        ol.highlighted = next((i for i, o in enumerate(ol.options) if not o.disabled), None)
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id)

    def action_close(self) -> None:
        self.dismiss(None)
