"""The Textual patches must not change what appears on screen."""

import re

from rich.color import Color, ColorSystem
from rich.console import Console
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

from nctui._textual_fixes import patch_strip_render

SGR = re.compile(r"\x1b\[([0-9;]*)m")


def screen_cells(ansi: str) -> list[tuple[str, tuple | None, tuple | None, bool]]:
    """What each character looks like: (char, fg, bg, bold), replaying the colour codes."""
    fg = bg = None
    bold = False
    out = []
    pos = 0
    for m in SGR.finditer(ansi + "\x1b[m"):
        out += [(ch, fg, bg, bold) for ch in ansi[pos:m.start()]]
        pos = m.end()
        codes = [int(c) for c in m.group(1).split(";") if c] or [0]
        i = 0
        while i < len(codes):
            c = codes[i]
            if c == 0:
                fg = bg = None
                bold = False
            elif c == 1:
                bold = True
            elif c == 22:
                bold = False
            elif c == 39:
                fg = None
            elif c == 49:
                bg = None
            elif c in (38, 48) and codes[i + 1] == 2:
                rgb = tuple(codes[i + 2:i + 5])
                fg, bg = (rgb, bg) if c == 38 else (fg, rgb)
                i += 4
            i += 1
    return out


def test_colour_runs_look_the_same_with_fewer_bytes():
    red, blue = Color.from_rgb(200, 0, 0), Color.from_rgb(0, 0, 200)
    segments = [
        Segment("▀", Style(color=red, bgcolor=blue)),
        Segment("▀", Style(color=red, bgcolor=red)),  # only the background changes
        Segment("▀▀", Style(color=blue, bgcolor=red)),
        Segment(" ", None),  # plain text: back to the terminal's defaults
        Segment("12", Style(color=red, bold=True)),  # an attribute: written in full
        Segment(" ", Style(bgcolor=blue)),
        Segment("x", Style()),
    ]
    console = Console(color_system="truecolor", force_terminal=True)
    plain = "".join(seg.text if seg.style is None else
                    seg.style.render(seg.text, color_system=ColorSystem.TRUECOLOR) for seg in segments)
    assert patch_strip_render()
    lean = Strip(segments).render(console)
    assert screen_cells(lean) == screen_cells(plain)
    assert len(lean) < len(plain)
    assert lean.endswith("\x1b[0m")  # leaves the terminal as it found it
