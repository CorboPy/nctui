import pytest

from nctui.data import Source
from nctui.reduce import AxisMap
from nctui.state import Viewer

from .conftest import make_pressure


@pytest.fixture
def viewer(cartesian):
    v = Viewer(Source(cartesian))
    v.select_var("theta")
    return v


def test_default_view_is_horizontal_map(viewer):
    assert viewer.kind == "map"
    assert (viewer.xdim, viewer.ydim) == ("projection_x_coordinate", "projection_y_coordinate")
    assert viewer.free_dims() == ["time", "full_levels"]
    # probe starts in the middle of the map, time and level at the start
    assert viewer.pos("projection_x_coordinate") == 3
    assert viewer.pos("time") == 0


def test_brackets_step_time_and_level(viewer):
    assert viewer.step_targets() == ("time", "full_levels")
    viewer.step(viewer.step_targets()[0], 1)
    viewer.step(viewer.step_targets()[1], 2)
    assert viewer.pos("time") == 1 and viewer.pos("full_levels") == 2
    # stepping clamps at the ends
    assert not viewer.step("time", -5) or viewer.pos("time") == 0
    assert viewer.pos("time") == 0


def test_cycle_views(viewer):
    assert viewer.cycle_view()
    assert (viewer.xdim, viewer.ydim) == ("projection_x_coordinate", "full_levels")
    assert viewer.step_targets() == ("time", "projection_y_coordinate")
    assert viewer.cycle_view()
    assert (viewer.xdim, viewer.ydim) == ("projection_y_coordinate", "full_levels")
    assert viewer.cycle_view()
    assert viewer.ydim == "projection_y_coordinate"


def test_level_carries_over_between_full_and_half_levels(viewer):
    viewer.set_pos("full_levels", 3)  # value 3.0
    viewer.select_var("rho")  # half levels 0.5, 1.5, ...
    assert viewer.pos("half_levels") in (2, 3)  # 2.5 or 3.5
    viewer.set_pos("half_levels", 1)  # 1.5
    viewer.select_var("theta")
    assert viewer.pos("full_levels") in (1, 2)


def test_view_kept_when_switching_variable(viewer):
    viewer.cycle_view()  # z-x section
    viewer.select_var("rho")
    assert (viewer.xdim, viewer.ydim) == ("projection_x_coordinate", "half_levels")


def test_profile_and_back(viewer):
    viewer.set_pos("projection_x_coordinate", 2)
    assert viewer.open_line("full_levels")
    assert viewer.kind == "line" and viewer.xdim == "full_levels"
    assert "projection_x_coordinate" in viewer.free_dims()
    # switching variable keeps the profile
    viewer.select_var("rho")
    assert viewer.kind == "line" and viewer.xdim == "half_levels"
    assert viewer.back()
    assert viewer.kind == "map"
    assert viewer.pos("projection_x_coordinate") == 2


def test_assign_axis_swaps(viewer):
    viewer.assign_axis("projection_y_coordinate", "x")
    assert (viewer.xdim, viewer.ydim) == ("projection_y_coordinate", "projection_x_coordinate")
    viewer.assign_axis("full_levels", "y")
    assert viewer.ydim == "full_levels"


def test_zoom_every_point_and_fit(tmp_path):
    from .conftest import make_cartesian

    v = Viewer(Source(make_cartesian(tmp_path / "big.nc", nt=1, nz=301, ny=4, nx=128)))
    v.select_var("theta")
    v.cycle_view()  # z-x section, 301 levels
    v.area = (151, 104)
    v.set_pos("full_levels", 150)
    assert v.every_point()
    assert v.window("full_levels") == (98, 202)
    assert v.window("projection_x_coordinate") == (0, 128)
    assert v.is_zoomed()
    # moving past the edge pans the window
    v.maps = {"full_levels": AxisMap(98, 104, 104, reverse=True),
              "projection_x_coordinate": AxisMap(0, 128, 128)}
    v.set_pos("full_levels", 201)
    v.move_probe(0, 1)
    assert v.pos("full_levels") == 202
    assert v.window("full_levels") == (99, 203)
    assert v.fit()
    assert v.window("full_levels") == (0, 301)


def test_zoom_in_out(viewer):
    viewer.zoom(2)
    a, b = viewer.window("projection_x_coordinate")
    assert b - a == 4
    viewer.zoom(0.5)
    assert viewer.window("projection_x_coordinate") == (0, 7)


def test_probe_moves_by_screen_direction(viewer):
    # y increases upwards on screen (reverse=True): "up" means a larger index
    viewer.maps = {"projection_x_coordinate": AxisMap(0, 7, 70), "projection_y_coordinate": AxisMap(0, 5, 50, True)}
    y0 = viewer.pos("projection_y_coordinate")
    viewer.move_probe(0, 1)
    assert viewer.pos("projection_y_coordinate") == y0 + 1
    x0 = viewer.pos("projection_x_coordinate")
    viewer.move_probe(1, 0)
    assert viewer.pos("projection_x_coordinate") == x0 + 1


def test_probe_steps_a_whole_pixel_when_downsampled(viewer):
    viewer.maps = {"projection_x_coordinate": AxisMap(0, 7, 3), "projection_y_coordinate": AxisMap(0, 5, 5, True)}
    viewer.set_pos("projection_x_coordinate", 0)
    viewer.move_probe(1, 0)
    assert viewer.pos("projection_x_coordinate") == 2  # 7 points on 3 pixels: one pixel ≈ 2 points


def test_colour_lock_and_manual(viewer):
    viewer.last_range = (1.0, 2.0)
    assert viewer.toggle_lock() == "locked"
    assert (viewer.colour.vmin, viewer.colour.vmax) == (1.0, 2.0)
    assert viewer.toggle_lock() == "auto"
    with pytest.raises(ValueError):
        viewer.set_manual_range(3, 1)
    viewer.set_manual_range(0, 5)
    assert viewer.colour.mode == "manual"
    # colour settings are per variable
    viewer.select_var("rho")
    assert viewer.colour.mode == "auto"


def test_equal_aspect_rules(viewer):
    assert viewer.equal_aspect()  # m and m
    viewer.cycle_view()
    assert not viewer.equal_aspect()  # m and levels
    viewer.options.aspect = "equal"
    assert viewer.equal_aspect()


def test_coastlines_only_for_lat_lon(viewer, latlon):
    assert not viewer.can_show_coastlines()
    v = Viewer(Source(latlon))
    v.select_var("air_temperature")
    assert v.can_show_coastlines()


def test_request_describes_the_slice(viewer):
    viewer.set_pos("time", 2)
    req = viewer.request()
    idx = req.index_dict()
    assert idx["time"] == 2
    assert idx["projection_x_coordinate"] == (0, 7)
    assert req.kind == "map"


def test_two_dimensional_variable_without_roles(tmp_path):
    import numpy as np
    import xarray as xr

    p = tmp_path / "plain.nc"
    xr.Dataset({"a": (("row", "col"), np.zeros((3, 4)))}).to_netcdf(p)
    v = Viewer(Source(p))
    v.select_var("a")
    assert (v.xdim, v.ydim) == ("col", "row")


def test_pressure_profile_points_down(tmp_path):
    v = Viewer(Source(make_pressure(tmp_path / "p.nc")))
    v.select_var("ta")
    assert (v.xdim, v.ydim) == ("lat", "plev")
    assert v.open_line("plev")
    assert not v.coord_up()


def test_centre_here_keeps_the_probe_on_its_point(latlon):
    v = Viewer(Source(latlon))
    v.select_var("air_temperature")
    v.set_pos("longitude", 10)  # 315°
    v.set_pos("latitude", 4)
    value = v.src.read("air_temperature", {"time": 0, "level_height": 0})[4, 10]
    assert v.centre_here() == ["longitude"]  # latitude only goes over the pole when asked
    lon = v.src.dims["longitude"]
    assert lon.values[v.pos("longitude")] == 315.0
    assert lon.label(v.pos("longitude")) == "-45"  # reads -180…180
    assert v.pos("longitude") in (5, 6)  # in the middle
    assert v.src.read("air_temperature", {"time": 0, "level_height": 0})[4, v.pos("longitude")] == value
    assert v.uncentre()
    assert v.pos("longitude") == 10


def test_equal_aspect_with_different_units_draws_square_cells(viewer):
    """x in metres against level numbers: no common scale, so each grid cell is square."""
    from nctui.frame import build_frame

    viewer.set_axes("projection_x_coordinate", "full_levels")
    viewer.options.aspect = "equal"
    f = build_frame(viewer.src, viewer.request(), 120, 40)
    xm, ym = f.maps["projection_x_coordinate"], f.maps["full_levels"]
    assert xm.out / xm.n == ym.out / ym.n


def test_centre_here_with_latitude_puts_the_probe_in_the_middle(latlon):
    v = Viewer(Source(latlon))
    v.select_var("air_temperature")
    v.set_pos("longitude", 2)  # 75°
    v.set_pos("latitude", 7)  # 60°
    value = v.src.read("air_temperature", {"time": 0, "level_height": 0})[7, 2]
    assert v.centring_axes(latitude=False) == ["longitude"]
    assert v.centre_here(latitude=True) == ["longitude", "latitude"]
    lat, lon = v.src.dims["latitude"], v.src.dims["longitude"]
    assert lat.label(v.pos("latitude")) == "60" and v.pos("latitude") == 9  # the middle of 18 rows
    assert lon.label(v.pos("longitude")) == "75"
    assert v.src.read("air_temperature", {"time": 0, "level_height": 0})[v.pos("latitude"), v.pos("longitude")] \
        == value
    assert v.uncentre()
    assert (v.pos("latitude"), v.pos("longitude")) == (7, 2)


# --------------------------------------------------------------------------- reducing over dimensions

T, Z, Y, X = "time", "full_levels", "projection_y_coordinate", "projection_x_coordinate"


def test_reducing_a_free_dimension_keeps_the_view(viewer):
    viewer.set_pos(T, 2)
    assert viewer.set_reduction(T, "mean")
    assert not viewer.set_reduction(T, "mean")  # nothing to change
    assert (viewer.kind, viewer.xdim, viewer.ydim) == ("map", X, Y)
    assert viewer.reduction(T) == "mean" and viewer.reductions() == ((T, "mean"),)
    assert viewer.live_dims() == [Z, Y, X] and viewer.free_dims() == [Z]
    req = viewer.request()
    assert req.reduce == ((T, "mean"),)
    assert [d for d, _ in req.index] == [Z, Y, X]  # no position along a collapsed dimension
    # restoring it puts it back where it was
    assert viewer.set_reduction(T, None)
    assert viewer.reductions() == () and viewer.pos(T) == 2
    assert viewer.request().reduce == ()


def test_a_reduced_dimension_cant_be_stepped_plotted_or_animated(viewer):
    viewer.set_reduction(T, "max")
    assert viewer.step_targets() == (Z, None)  # [ ] now step the level
    assert viewer.sweep_dim() == Z
    assert not viewer.step(T, 1) and not viewer.step(T, 1, wrap=True) and not viewer.set_pos(T, 3)
    assert viewer.pos(T) == 0
    assert not viewer.assign_axis(T, "x") and not viewer.open_line(T)
    assert (viewer.kind, viewer.xdim, viewer.ydim) == ("map", X, Y)
    with pytest.raises(ValueError, match="restore it first"):
        viewer.set_axes(T, Z)
    with pytest.raises(ValueError, match="restore it first"):
        viewer.set_axes(Z, T)
    for _ in range(4):
        viewer.cycle_view()
        assert T not in viewer.plotted()


def test_reducing_a_plotted_dimension_changes_the_view(viewer):
    viewer.set_reduction(X, "mean")  # the "zonal" mean of a y–x map: a z–y section
    assert (viewer.kind, viewer.xdim, viewer.ydim) == ("map", Y, Z)
    assert viewer.free_dims() == [T]
    assert viewer.available_views() == [(Y, Z)]
    viewer.set_reduction(Y, "mean")  # time and level left: no preset view, the last two dimensions
    assert (viewer.kind, viewer.xdim, viewer.ydim) == ("map", Z, T)
    viewer.set_reduction(Z, "mean")  # the domain-mean time series
    assert (viewer.kind, viewer.xdim, viewer.ydim) == ("line", T, None)
    assert viewer.free_dims() == [] and viewer.step_targets() == (None, None)
    viewer.set_reduction(T, "mean")  # nothing left: one number
    assert (viewer.kind, viewer.xdim, viewer.ydim) == ("scalar", None, None)
    assert viewer.request().index == () and len(viewer.request().reduce) == 4
    # coming back: a line when one dimension is left, a map with two
    viewer.set_reduction(Z, None)
    assert (viewer.kind, viewer.xdim) == ("line", Z)
    viewer.set_reduction(X, None)
    assert (viewer.kind, viewer.xdim, viewer.ydim) == ("map", X, Z)


def test_restoring_keeps_a_view_that_still_stands(viewer):
    viewer.set_reduction(X, "mean")
    viewer.set_reduction(X, None)
    assert (viewer.kind, viewer.xdim, viewer.ydim) == ("map", Y, Z)  # x is back, as a free dimension
    assert viewer.free_dims() == [T, X]


def test_reducing_a_profiles_own_dimension_goes_back_to_the_map(viewer):
    viewer.open_line(Z)
    viewer.set_reduction(T, "mean")  # a free dimension: still the profile
    assert (viewer.kind, viewer.xdim) == ("line", Z) and viewer.stack
    viewer.set_reduction(Z, "mean")  # the vertical mean, on the map the profile came from
    assert (viewer.kind, viewer.xdim, viewer.ydim) == ("map", X, Y) and not viewer.stack


def test_back_from_a_profile_when_the_maps_dimension_was_reduced(viewer):
    viewer.open_line(Z)
    viewer.set_reduction(X, "mean")
    assert viewer.back()
    assert viewer.kind == "map" and X not in viewer.plotted()


def test_reductions_carry_over_to_other_variables(viewer):
    viewer.set_reduction(T, "mean")
    viewer.set_reduction(Z, "max")
    viewer.select_var("rho")  # on half_levels: its vertical dimension is reduced the same way
    assert viewer.reductions() == ((T, "mean"), ("half_levels", "max"))
    assert (viewer.kind, viewer.xdim, viewer.ydim) == ("map", X, Y)
    viewer.select_var("forecast_period")  # only has time: one number
    assert viewer.kind == "scalar" and viewer.reductions() == ((T, "mean"),)
    viewer.select_var("theta")
    assert viewer.reductions() == ((T, "mean"), (Z, "max"))
    assert viewer.kind == "map"


def test_reductions_are_part_of_the_animation_range_key(viewer):
    plain = viewer.anim_range_key()
    viewer.set_reduction(T, "mean")
    mean = viewer.anim_range_key()
    assert mean[1] == Z  # the animation now runs along the level
    viewer.set_reduction(T, "max")
    assert len({plain, mean, viewer.anim_range_key()}) == 3
    viewer.anim_ranges[mean] = (0.0, 1.0)
    assert viewer.request().anim_range is None  # the mean's range isn't the max's
    viewer.set_reduction(T, "mean")
    assert viewer.request().anim_range == (0.0, 1.0)


def test_unknown_statistic_or_dimension(viewer):
    with pytest.raises(ValueError):
        viewer.set_reduction(T, "median")
    with pytest.raises(ValueError):
        viewer.set_reduction("half_levels", "mean")


def test_snapshot_and_restore(viewer):
    viewer.set_pos(Z, 3)
    viewer.zoom(2)
    snap = viewer.snapshot()
    before = viewer.request()
    viewer.set_reduction(X, "mean")
    viewer.set_reduction(T, "sum")
    viewer.set_pos(T, 1)
    viewer.select_var("rho")
    viewer.restore(snap)
    assert viewer.request() == before
    assert viewer.reductions() == ()


def test_zoom_and_centring_dont_change_what_is_reduced(latlon):
    v = Viewer(Source(latlon))
    v.select_var("air_temperature")
    v.set_reduction("time", "mean")
    key = v.src.reduce_key("air_temperature", v.request().index_dict(), v.request().reduce)
    v.zoom(2)
    v.centre_here(latitude=True)
    req = v.request()
    assert req.reduce == (("time", "mean"),)
    assert v.src.reduce_key("air_temperature", req.index_dict(), req.reduce) == key
    # reducing the centred longitude takes it off the plot; its centring stays for when it comes back
    v.set_reduction("longitude", "mean")
    assert (v.xdim, v.ydim) == ("latitude", "level_height")
    assert v.src.centre("longitude") is not None
