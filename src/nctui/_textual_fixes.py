"""Two small changes to Textual that keep animations smooth.

1. Garbage.
``Strip.divide`` caches its result inside the strip. When nothing needs cutting
the result is ``[strip]``, so the strip refers to itself: a reference cycle. The
compositor divides every line of every widget on every refresh, so while a
plot animates each frame leaves thousands of segments that only Python's full
garbage collection can free, and that collection pauses the app for tens of
milliseconds a few times a second.

Returning ``[strip]`` without caching it gives the same result and lets
reference counting free old frames straight away.

2. Bytes. Textual writes every segment as its full style, then a reset:
``ESC[38;2;r;g;b;48;2;r;g;bm▀ESC[0m`` for each cell of a plot, about 42 bytes. A
large plot at 1:1 is ~600 KB a frame, more than a terminal can draw 20 times a
second. Runs of colour-only segments only need the colour that changed, and no
reset in between, which cuts the plot's output by about a quarter.
"""

from __future__ import annotations

from rich.color import ColorSystem
from textual.strip import Strip


def patch_strip_divide() -> bool:
    """Apply the fix (once). Returns False if this Textual's ``Strip`` looks different."""
    original = Strip.divide
    if getattr(original, "_nctui_patched", False):
        return True
    if not hasattr(Strip, "cell_length"):
        return False

    def divide(self: Strip, cuts):
        cell_length = self.cell_length
        kept = [cut for cut in cuts if cut <= cell_length]
        if kept == [cell_length]:
            return [self]
        return original(self, kept)

    divide._nctui_patched = True  # type: ignore[attr-defined]
    divide.__doc__ = original.__doc__
    Strip.divide = divide  # type: ignore[method-assign]
    return True


def patch_strip_render() -> bool:
    """Write colour-only segments as colour changes, without a reset after each (once).

    Returns False if this Textual's ``Strip`` looks different (then nothing changes).
    """
    original = Strip.render
    if getattr(original, "_nctui_patched", False):
        return True
    if not hasattr(Strip, "_render_cache") and "_render_cache" not in getattr(Strip, "__slots__", ()):
        probe = Strip([])
        if not hasattr(probe, "_render_cache"):
            return False

    def render(self: Strip, console) -> str:
        if self._render_cache is not None:
            return self._render_cache
        color_system = console._color_system or ColorSystem.TRUECOLOR
        full = self.render_style
        out: list[str] = []
        fg = bg = None  # what the terminal is set to by the colour-only segments so far
        dirty = False
        for text, style, _ in self._segments:
            if style is not None and not style._set_attributes and style._link is None:
                codes: list[str] = []
                if not dirty or style.color != fg:
                    fg = style.color
                    codes.extend(fg.downgrade(color_system).get_ansi_codes(True) if fg else ("39",))
                if not dirty or style.bgcolor != bg:
                    bg = style.bgcolor
                    codes.extend(bg.downgrade(color_system).get_ansi_codes(False) if bg else ("49",))
                if codes:
                    out.append(f"\x1b[{';'.join(codes)}m")
                out.append(text)
                dirty = True
                continue
            if dirty:  # back to the terminal's defaults before anything else
                out.append("\x1b[0m")
                dirty = False
            out.append(text if style is None else full(style, text, color_system=color_system))
        if dirty:
            out.append("\x1b[0m")
        self._render_cache = "".join(out)
        return self._render_cache

    render._nctui_patched = True  # type: ignore[attr-defined]
    render.__doc__ = original.__doc__
    Strip.render = render  # type: ignore[method-assign]
    return True
