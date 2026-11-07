import numpy as np

from nctui import coastlines
from nctui.data import Source
from nctui.frame import build_frame
from nctui.render import BG, NAN, colour_indices, format_ticks, halfblock_rows, nice_ticks
from nctui.state import Viewer


def test_colour_indices_span_the_map():
    idx = colour_indices(np.array([0.0, 0.5, 1.0, np.nan]), 0.0, 1.0)
    assert list(idx) == [0, 128, 255, NAN]


def test_colour_indices_clip_and_log():
    idx = colour_indices(np.array([-5.0, 5.0]), 0.0, 1.0)
    assert list(idx) == [0, 255]
    idx = colour_indices(np.array([1.0, 10.0, 100.0, -1.0]), 1.0, 100.0, log=True)
    assert list(idx) == [0, 128, 255, 0]


def test_halfblock_two_pixels_per_cell():
    idx = np.array([[1, 2], [3, 4]], dtype=np.int16)
    rows = halfblock_rows(idx, "viridis")
    assert len(rows) == 1
    text = "".join(s.text for s in rows[0])
    assert text == "▀▀"


def test_halfblock_odd_height_pads_the_top():
    idx = np.array([[1], [2], [3]], dtype=np.int16)
    rows = halfblock_rows(idx, "viridis")
    assert len(rows) == 2
    assert rows[0][0].text == "▄"  # top half empty, bottom half is pixel 0


def test_nice_ticks():
    assert nice_ticks(0, 10, 5) == [0, 2, 4, 6, 8, 10]
    assert nice_ticks(350.0, 350.07, 4) == [350.0, 350.02, 350.04, 350.06]


def test_format_ticks_resolves_close_values():
    assert format_ticks([350.0, 350.02, 350.04]) == ["350.00", "350.02", "350.04"]
    assert format_ticks([-100000.0, 0.0, 100000.0]) == ["-100000", "0", "100000"]
    assert format_ticks([-8e-08, 0.0, 8e-08]) == ["-8e-08", "0", "8e-08"]


def test_coastlines_rasterise_on_global_grid():
    lon = np.linspace(1.25, 358.75, 144)
    lat = np.linspace(89, -89, 90)
    mask = coastlines.rasterise(lon, lat)
    assert mask.shape == (90, 144)
    assert 0.02 < mask.mean() < 0.3
    # the middle of the Pacific (about 180E, 0N) has no coast
    assert not mask[44:46, 70:74].any()


def test_coastlines_wrap_to_minus_180_convention():
    lon = np.linspace(-178.75, 178.75, 144)
    lat = np.linspace(89, -89, 90)
    assert coastlines.rasterise(lon, lat).any()


def test_map_frame(cartesian):
    src = Source(cartesian)
    v = Viewer(src)
    v.select_var("theta")
    v.set_pos("time", 1)
    f = build_frame(src, v.request(), 80, 30)
    assert f.error is None
    assert len(f.lines) <= 30
    assert all(sum(s.cell_length for s in line) == 80 for line in f.lines)
    assert f.vrange == (1000 + 0 + 0 + 0, 1000 + 0 + 40 + 6)
    # probe in the middle of the map: x=3, y=2
    assert f.probe_value == 1000 + 20 + 3
    assert "time 2000-01-01 00:30" in f.header
    assert "1:1" in f.status or "×" in f.status


def test_auto_diverging_colours(cartesian):
    src = Source(cartesian)
    v = Viewer(src)
    v.select_var("dtheta")
    f = build_frame(src, v.request(), 80, 30)
    assert f.colour is not None
    assert f.colour.cmap == "RdBu_r" and f.colour.symmetric
    assert f.vrange[0] == -f.vrange[1]


def test_line_frame(cartesian):
    src = Source(cartesian)
    v = Viewer(src)
    v.select_var("theta")
    v.open_line("full_levels")
    f = build_frame(src, v.request(), 60, 20)
    assert f.error is None
    assert f.line_fig is not None and f.line_fig.vertical
    assert "profile along full_levels" in f.header
    assert "6 points" in f.status


def test_small_window_message(cartesian):
    src = Source(cartesian)
    v = Viewer(src)
    v.select_var("theta")
    f = build_frame(src, v.request(), 20, 5)
    assert "bigger" in "".join(s.text for line in f.lines for s in line)


def test_nan_counted(latlon):
    src = Source(latlon)
    v = Viewer(src)
    v.select_var("air_temperature")
    f = build_frame(src, v.request(), 80, 30)
    assert "missing 1" in f.stats


def test_coastline_overlay_frame(latlon):
    src = Source(latlon)
    v = Viewer(src)
    v.select_var("air_temperature")
    v.options.coastlines = True
    f = build_frame(src, v.request(), 100, 40)
    assert f.error is None


def test_unused_bg_constant():
    assert BG == -1


def test_auto_colour_waits_for_a_varying_slice(tmp_path):
    import numpy as np
    import xarray as xr

    w = np.zeros((2, 3, 4), dtype=np.float32)
    w[:, 1:] = np.linspace(-2, 2, 4)  # zero at the surface, both signs above
    p = tmp_path / "w.nc"
    xr.Dataset({"w": (("z", "y", "x"), w.transpose(1, 0, 2))}).to_netcdf(p)
    src = Source(p)
    v = Viewer(src)
    v.select_var("w")
    assert build_frame(src, v.request(), 80, 30).colour is None  # constant slice: undecided
    v.set_pos("z", 1)
    f = build_frame(src, v.request(), 80, 30)
    assert f.colour is not None and f.colour.cmap == "RdBu_r"


def test_time_ticks_line_up_with_the_columns(cartesian):
    """A time–height section: every x tick sits in the middle of a time step's column."""
    from nctui.data import Source
    from nctui.frame import build_frame
    from nctui.state import Viewer

    v = Viewer(Source(cartesian))
    v.select_var("theta")
    v.set_axes("time", "full_levels")
    v.set_pos("time", 3)  # keep the probe marker off the ticks being checked
    f = build_frame(v.src, v.request(), 100, 30)
    xm = f.maps["time"]
    axis = "".join(seg.text for seg in f.lines[-3])
    ticks = [i - f.image_col for i, ch in enumerate(axis) if ch == "┬"]
    centres = [(a + b) // 2 for a, b in (xm.pixel_span(i) for i in range(xm.n))]
    assert ticks and all(t in centres for t in ticks)


def test_degree_ticks_land_on_the_pole_and_equator():
    from nctui.render import degree_ticks

    assert degree_ticks(-10, 170, 7) == [0, 30, 60, 90, 120, 150]
    assert degree_ticks(-180, 180, 5) == [-180, -90, 0, 90, 180]
    assert degree_ticks(50.2, 50.3, 4) == nice_ticks(50.2, 50.3, 4)


def test_wrapped_latitude_keeps_the_longitudes_and_marks_the_seam(latlon):
    v = Viewer(Source(latlon))
    v.select_var("air_temperature")
    v.centre_on("longitude", 0.0)
    plain = build_frame(v.src, v.request(), 100, 30)
    v.centre_on("latitude", 90.0)
    v.options.coastlines = True
    f = build_frame(v.src, v.request(), 100, 30)
    text = ["".join(s.text for s in line) for line in f.lines]
    assert text[-2] == "".join(s.text for s in plain.lines[-2])  # the same longitude labels
    assert "±90" in "".join(text[f.image_row:f.image_row + 20])  # where the poles meet


def test_line_along_a_wrapped_latitude_reads_true_latitudes(latlon):
    v = Viewer(Source(latlon))
    v.select_var("air_temperature")
    v.centre_on("latitude", 90.0)
    v.open_line("latitude")
    f = build_frame(v.src, v.request(), 100, 24)
    labels = "".join(s.text for s in f.lines[-2]).split()
    assert "±90" in labels
    assert labels.index("±90") not in (0, len(labels) - 1)
    assert all(-90 < float(t) < 90 for t in labels if t != "±90")
