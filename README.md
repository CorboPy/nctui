# nctui

A fast terminal viewer for weather and climate model netCDF output. Open a file and
you get a plot straight away. Flick through variables, times and levels from the
keyboard (or the mouse), probe values, pull out vertical profiles and time series,
and save a PNG, all without leaving the terminal.

```
nctui run_output.nc
```

<img width="1512" height="826" alt="nctui_display" src="https://github.com/user-attachments/assets/7c182c15-1d48-4621-8790-267dfc0d4dc1" />


## Install

nctui uses [pixi](https://pixi.sh). The environment builds on macOS (Apple
silicon) and Linux x86-64.

```bash
pixi install
```

```bash
pixi run nctui example_data/lfric_diag.nc
```

Or activate the environment with `pixi shell` and run `nctui` directly. Because the
package is a normal Python project, `pip install .` also works if you'd rather
manage the dependencies yourself (Python ≥ 3.12, textual, netCDF4, cftime, numpy,
matplotlib).

## Using it

```
nctui FILE [FILE...]        open; several files are joined along their time dimension
nctui FILE -v theta         open straight on a variable
```

The screen has three panes. Move between them with `tab`, or click.

- **Variables** (left, top): variables grouped by number of dimensions. Moving the
  highlight plots that variable straight away. `/` searches by name, long_name or
  standard_name.
- **Dimensions** (left, bottom): where you are along each dimension. `←→` steps
  the highlighted one, wrapping round from the last step to the first (on a plotted
  dimension it moves the probe, also wrapping round), and `x` / `y` put it on the
  plot's axes.
- **Plot** (right): one plot at a time. The arrow keys move a probe that reads out the
  value at that grid point. `enter` opens the vertical profile, time series or a line
  through the probe; `esc` goes back.

When you change variable, nctui keeps your place: the same time, the nearest level
(so `full_levels` and `half_levels` line up), the same probe point and the same
view.

### Keys

| Key | Action |
|---|---|
| `[` `]` | step time (wraps round from the last step to the first) |
| `{` `}` | step the vertical level (wraps round) |
| `v` | cycle views: map → section (z–x) → section (z–y) |
| `t` | swap the plot axes |
| `space`, `<` `>` | play / pause animation, slower / faster |
| `enter` | actions for whatever is selected (profile, time series, average over a dimension, …) |
| `+` `-` | zoom in / out around the probe; or drag a box with the mouse |
| `1` / `0` | show every data point around the probe / fit the whole field |
| `f` | full-width plot (hide the left panes) |
| `c` / `m` / `r` | colours popup / next colour map / lock the colour range |
| `p` | plot settings: axes, aspect, centre (wrap longitude / latitude), how points are combined, coastlines |
| `o` | coastlines on lat/lon maps (off by default; not every planet is Earth) |
| `g` | go to a value: a date, a height, `#index`, or a range like `0:2000` |
| `e` | save the current view (PNG/PDF/SVG), or an animation of it (MP4/GIF) |
| `i` / `?` / `q` | attributes / help / quit |

Popups that change the plot (colours, plot settings) apply the change as soon as you
make it, so you can see the result behind them.

### Colour range

The colours popup (`c`) has four ways to set the range:

- **auto**: each frame is fitted to its own min and max, so the same colour can mean
  different values as you step through time.
- **locked**: keeps the range on screen now (also `r`).
- **min/max: fit animation**: the min and max over every step of the dimension an
  animation runs along (time, unless you animate something else), for the view on
  screen. The other dimensions stay where they are: on level 20 it scans level 20
  at every time. Change the level or zoom and it scans again. Use this to animate
  or save an animation with a fixed colour bar.
- **manual**: type a min and max; it applies on `enter`, `tab` or clicking away.

### Centring a periodic axis

Longitude that goes all the way round can be wrapped so another value sits in the
middle: press `enter` on the plot and choose *Centre here*, or type a value
under *centre on* in Plot settings (`p`). A 0–360° grid centred on 0 reads -180…180,
so the UK is no longer split across the edges. The x and y of a Cartesian grid are
taken to be periodic (as bi-periodic model domains are), so they can be centred the
same way.

Latitude can be centred when the grid reaches both poles. It is mirrored at the
poles, at the same longitudes, and the plot shows every row twice. Centred on 90,
it runs from the south pole up to the north pole in the middle, then back down to
the south pole at the top. Centred on -90 it mirrors about the south pole instead:
the north pole is at both the top and the bottom. *Centre here… → longitude + latitude*
in the enter menu centres both on the probe.

### Averaging over a dimension

Any dimension can be collapsed with a statistic, for a zonal mean, a time mean, a
vertical mean or a domain-mean time series. Press `enter` on the plot, or on a
dimension in the Dimensions pane, and choose *Reduce* and the dimension, then *Mean*,
*Minimum*, *Maximum* or *Sum*.

- The Dimensions pane shows a reduced dimension as `time  mean`. It has no
  position, so it can't be stepped, animated or put on an axis. `enter` on it
  restores it (or changes the statistic), and it comes back where it was.
- Reducing a dimension that is on the plot changes the view to what is left: the
  zonal mean of a lat–lon map is a latitude–level section, and with latitude and
  longitude both reduced you get a time series (a map against level if the
  variable has levels). Reduce everything and you get the one number.
- Reductions stay when you change variable, like your place does: a vertical mean
  over `full_levels` is a vertical mean over `half_levels` too.
- Missing values are ignored, and a point with nothing but missing values stays
  missing.
- A mean over a latitude in degrees is weighted by cos(latitude). The status line
  says which was used: `cos(lat)-weighted mean` or `unweighted mean`. Nothing
  else is weighted: time steps count equally, whatever their length, and a
  vertical mean or sum is over levels, not mass or height.
- Dimensions reduced with the same statistic are collapsed together, so a mean
  over latitude and longitude is one area-weighted mean of every valid point.
  Different statistics are applied from the variable's last dimension to its first
  (so space before time): *Mean* over longitude with *Maximum* over time is the
  largest zonal mean.
- A dimension is always reduced over its whole length, each point once, however
  the plot is zoomed or centred. Zoom, the probe, line plots, centring and
  *min/max: fit animation* all work on the reduced field.

The file is read a block at a time, off the screen's thread. A reduction that takes
a moment shows its progress in the top bar (`mean over time 40% · esc cancels`);
`esc` stops it and goes back to the plot you had. The result is kept, so zooming,
scrolling and centring afterwards are instant, but each new level (or time) of a
reduced view is read when you first go there.

### Saving a figure or an animation

Get the plot looking the way you want, then press `e`. The dialog is filled in
from the current view; you can change the file name, title, size and dpi. The image
is drawn by matplotlib with your colour map, range, view, zoom and coastlines, using
**every data point**, not the reduced version on screen. A mean (or other
statistic) over a dimension is worked out from the file's data too, and the title
and file name say what was reduced. Name the file `.pdf` or `.svg` for those
formats.

Choose *animation* in the same dialog to save every step along the animated dimension as a movie, at your playback speed.
You can pick the first and last frame and the frame rate. Each frame's title gets
its own time (or level). Pick *mp4* or *gif* under *format* (or just type the
file name's extension). mp4 needs `ffmpeg` on your `PATH`; gif always works.
With the *auto* colour range the colour bar changes every frame. For one fixed range,
choose *min/max: fit animation* in the colours popup first.

### Terminal resolution

A terminal cell holds two roughly square pixels (nctui draws with the `▀` character
in two colours), so a full-screen laptop terminal gives the plot about 180 × 100
pixels. However, if there are more
levels than the screen has pixel rows, nctui handles this as follows:

- The status line under the plot always shows how the data was fitted:
  `full_levels 301→104 ÷2.9 (mean)` means about three levels share each pixel row;
  `1:1` means every point is shown.
- Plot settings (`p`) choose how points that share a pixel are combined. *Mean*
  is the default; *max*, *min* and *max |value|* keep thin features such as a
  one-level cloud layer from being averaged away.
- `1` shows every point around the probe, and moving the probe past an edge scrolls
  the view. Dragging a box, or typing a range in Go to (`g`), zooms in.
- The probe readout and profile plots always use the original data.
- A smaller terminal font (`Cmd -` in iTerm2) gives you more pixels straight away.

More pixels cost more to draw. A 128 × 300 plot at 1:1 is about half a megabyte of
terminal output per frame, and at 20 fps that's more than iTerm2 can draw: the
terminal, not nctui, is what uses the CPU. When the terminal falls behind, nctui
skips animation steps instead of queueing frames, so it stays responsive, and the
top bar shows the rate you're really getting (`▶ playing time · 8 of 20 fps`).
Slowing down (`<`) or a smaller plot eases the load.

### Quick look without the TUI

```bash
nctui plot FILE -v theta --at time=#3 --at full_levels=120.5
```

prints the same plot to stdout and exits, which is handy in scripts and job logs.
`--x`/`--y` choose the axes (give only `--x` for a line plot), `--cmap` picks a
colour map, and `--width`/`--height` set the size.

`--mean DIM`, `--min DIM`, `--max DIM` and `--sum DIM` reduce over a dimension, and
can be repeated:

```bash
nctui plot FILE -v tas --mean time                 # time-mean map
```

```bash
nctui plot FILE -v tas --mean lat --mean lon       # global-mean time series
```

## Development

```bash
pixi run test
```

```bash
pixi run lint
```

- `src/nctui/data.py` reads files (netCDF4, lazily, with a slice cache).
- `state.py` holds what's being viewed and every way it can change.
- `reduce.py` and `render.py` fit data to the screen and draw it.
- `frame.py` ties them together off the UI thread.
- `app.py`, `widgets/` and `screens/` make up the Textual UI.

`scripts/build_colormaps.py` regenerates the colour-map tables from matplotlib, and
`scripts/build_coastlines.py` rebuilds the bundled coastline file from Natural
Earth's `ne_110m_coastline.zip`.

## Data credits

Coastlines are from [Natural Earth](https://www.naturalearthdata.com/) (public
domain). The colour maps come from [matplotlib](https://matplotlib.org/stable/gallery/color/colormap_reference.html).

## Licence

MIT; see [LICENSE](LICENSE).
