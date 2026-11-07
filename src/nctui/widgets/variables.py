"""The variable tree: variables grouped by number of dimensions, with units."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.binding import Binding
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from ..data import Source

if TYPE_CHECKING:
    from ..app import NctuiApp

GROUP_ORDER = ["4D", "3D", "2D", "1D", "0D", "5D+", "other"]


def group_of(src: Source, name: str) -> str:
    v = src.vars[name]
    if not v.plottable:
        return "other"
    if v.ndim >= 5:
        return "5D+"
    return f"{v.ndim}D"


class VariableTree(Tree[str]):
    BINDINGS = [
        Binding("enter", "app.variable_menu", "actions", key_display="enter"),
        Binding("space", "app.play", "play", show=False),
    ]

    def __init__(self, source: Source, **kwargs):
        super().__init__("variables", **kwargs)
        self.src = source
        self.show_root = False
        self.guide_depth = 2
        self.filter_text = ""
        self._leaves: dict[str, TreeNode[str]] = {}

    @property
    def nctui(self) -> NctuiApp:
        return self.app  # type: ignore[return-value]

    def rebuild(self, filter_text: str = "", select: str | None = None) -> None:
        self.filter_text = filter_text
        self.clear()
        self._leaves = {}
        groups: dict[str, list[str]] = {}
        for name, v in self.src.vars.items():
            if filter_text and not v.matches(filter_text):
                continue
            groups.setdefault(group_of(self.src, name), []).append(name)
        width = max((len(n) for names in groups.values() for n in names), default=8)
        width = min(width, 22)
        for g in GROUP_ORDER:
            names = groups.get(g)
            if not names:
                continue
            label = Text(g, style="bold")
            if g not in ("other", "0D"):
                dims = self._common_dims(names)
                if dims:
                    label.append(f"  {dims}", style="dim")
            node = self.root.add(label, expand=g != "other")
            for name in names:
                v = self.src.vars[name]
                text = Text(name.ljust(width)[:width] if len(name) <= width else name[: width - 1] + "…")
                if v.units:
                    text.append(f" {v.units}", style="dim")
                self._leaves[name] = node.add_leaf(text, data=name)
        if select is not None:
            self.highlight(select)

    def _common_dims(self, names: list[str]) -> str:
        roles = {tuple(self.src.dims[d].role for d in self.src.vars[n].dims) for n in names}
        if len(roles) == 1:
            return "·".join(next(iter(roles)))
        return ""

    def highlight(self, name: str) -> None:
        node = self._leaves.get(name)
        if node is not None:
            self.move_cursor(node)

    def first_variable(self) -> str | None:
        return next(iter(self._leaves), None)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        event.stop()
        name = event.node.data
        if name is not None and self.src.vars[name].plottable:
            self.nctui.schedule_select(name)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        event.stop()
