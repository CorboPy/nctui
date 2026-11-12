"""Command line: ``nctui FILE...`` opens the viewer; ``nctui plot FILE...`` prints one plot."""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import sys

from . import __version__


def _open(files: list[str]):
    from .data import Source, SourceError

    try:
        return Source(files)
    except SourceError as exc:
        print(f"nctui: {exc}", file=sys.stderr)
        sys.exit(1)


def _check_var(src, name: str | None) -> None:
    if name is None or name in src.vars:
        return
    close = difflib.get_close_matches(name, list(src.vars), n=5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    print(f"nctui: no variable '{name}' in {src.title}.{hint}", file=sys.stderr)
    sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "plot" and not os.path.exists("plot"):
        return plot_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="nctui",
        description="Rapid terminal viewer for weather and climate model netCDF output. "
        "Several files are joined along their time dimension. "
        "Use 'nctui plot FILE -v VAR' to print a single plot instead.",
    )
    parser.add_argument("files", nargs="+", metavar="FILE", help="netCDF file(s)")
    parser.add_argument("-v", "--var", help="variable to show first")
    parser.add_argument("--version", action="version", version=f"nctui {__version__}")
    args = parser.parse_args(argv)
    src = _open(args.files)
    _check_var(src, args.var)

    from .app import NctuiApp

    NctuiApp(src, args.var).run()
    return 0


def plot_main(argv: list[str]) -> int:
    from .data import STATS

    parser = argparse.ArgumentParser(
        prog="nctui plot",
        description="Print one plot to stdout and exit (a quick look for scripts and logs).",
    )
    parser.add_argument("files", nargs="+", metavar="FILE")
    parser.add_argument("-v", "--var", help="variable (default: the first one)")
    parser.add_argument("--at", action="append", default=[], metavar="DIM=VALUE",
                        help="position along a dimension: a coordinate value, a date, or #index "
                             "(repeatable)")
    for stat in STATS:
        weighted = "; cos(lat)-weighted over a latitude in degrees" if stat == "mean" else ""
        parser.add_argument(f"--{stat}", action="append", default=[], metavar="DIM",
                            help=f"collapse a dimension to its {stat}, ignoring missing values{weighted} "
                                 "(repeatable)")
    parser.add_argument("--x", metavar="DIM", help="dimension on the x axis")
    parser.add_argument("--y", metavar="DIM", help="dimension on the y axis (omit with --x for a line plot)")
    parser.add_argument("--cmap", help="colour map name")
    parser.add_argument("--width", type=int, help="width in characters (default: terminal width)")
    parser.add_argument("--height", type=int, help="height in characters (default: 30)")
    parser.add_argument("--no-color", action="store_true", help="plain text output")
    args = parser.parse_args(argv)

    from rich.console import Console
    from rich.text import Text

    from . import colormaps
    from .frame import build_frame
    from .state import Viewer

    src = _open(args.files)
    _check_var(src, args.var)
    name = args.var or next((v.name for v in src.plottable_vars() if v.ndim >= 2), None) \
        or next((v.name for v in src.plottable_vars()), None)
    if name is None:
        print("nctui: no plottable variables", file=sys.stderr)
        return 1
    viewer = Viewer(src)
    viewer.select_var(name)
    try:
        chosen: dict[str, str] = {}
        for stat in STATS:
            for dim in getattr(args, stat):
                if chosen.setdefault(dim, stat) != stat:
                    raise ValueError(f"{dim}: give one statistic, not --{chosen[dim]} and --{stat}")
                viewer.set_reduction(dim, stat)
        if args.x:
            viewer.set_axes(args.x, args.y)
        for item in args.at:
            dim, sep, value = item.partition("=")
            if not sep or dim not in src.dims:
                raise ValueError(f"--at {item!r}: use DIM=VALUE with a dimension of {name}")
            if viewer.reduction(dim) is not None:
                raise ValueError(f"--at {item!r}: {dim} is reduced ({viewer.reduction(dim)}), it has no position")
            viewer.set_pos(dim, src.dims[dim].find(value))
    except ValueError as exc:
        print(f"nctui: {exc}", file=sys.stderr)
        return 1
    if args.cmap:
        if args.cmap not in colormaps.NAMES:
            print(f"nctui: unknown colour map; choose from {', '.join(colormaps.NAMES)}", file=sys.stderr)
            return 1
        viewer.colour.cmap = args.cmap
        viewer.colour.decided = True
    term = shutil.get_terminal_size((100, 32))
    width = args.width or term.columns
    height = args.height or 30
    frame = build_frame(src, viewer.request(), width, height)
    console = Console(no_color=args.no_color, highlight=False, width=width)
    v = src.vars[name]
    title = " · ".join(p for p in (v.name, v.long_name if v.long_name != v.name else "", v.units) if p)
    console.print(Text(title, style="bold"))
    console.print(Text(frame.header))
    for line in frame.lines:
        console.print(Text.assemble(*[(s.text, s.style) for s in line]), crop=True, overflow="crop",
                      no_wrap=True)
    console.print(Text(frame.stats, style="dim"))
    console.print(Text(frame.status, style="dim"))
    return 1 if frame.error else 0


if __name__ == "__main__":
    sys.exit(main())
