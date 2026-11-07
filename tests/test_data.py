import numpy as np
import pytest

from nctui.data import OTHER, TIME, Source, SourceError, X, Y, Z, classify_dim, format_lead, format_leads, format_number

from .conftest import EXAMPLE_DATA, make_cartesian, make_pressure


@pytest.mark.parametrize(
    "name, attrs, role",
    [
        ("time", {}, TIME),
        ("time_instant", {"axis": "T"}, TIME),
        ("t", {"units": "days since 2000-01-01"}, TIME),
        ("full_levels", {}, Z),
        ("half_levels", {}, Z),
        ("level_height", {"axis": "Z"}, Z),
        ("plev", {"units": "Pa"}, Z),
        ("model_level_number", {}, Z),
        ("projection_y_coordinate", {"units": "m"}, Y),
        ("latitude", {"units": "degrees_north"}, Y),
        ("lat", {}, Y),
        ("projection_x_coordinate", {"units": "m"}, X),
        ("longitude", {"units": "degrees_east"}, X),
        ("bnds", {}, OTHER),
        ("realization", {}, OTHER),
    ],
)
def test_classify_dim(name, attrs, role):
    assert classify_dim(name, attrs) == role


def test_open_cartesian(cartesian):
    src = Source(cartesian)
    assert src.time_dim == "time"
    assert src.dims["full_levels"].role == Z
    assert src.dims["projection_x_coordinate"].role == X
    assert src.dims["projection_y_coordinate"].role == Y
    assert {v.name for v in src.plottable_vars()} == {"theta", "rho", "dtheta", "forecast_period"}
    assert not src.vars["time_bnds"].plottable


def test_read_slice_and_window(cartesian):
    src = Source(cartesian)
    a = src.read("theta", {"time": 2, "full_levels": 3})
    assert a.shape == (5, 7)
    assert a.dtype == np.float64
    assert a[4, 6] == 2000 + 300 + 40 + 6
    w = src.read("theta", {"time": 1, "full_levels": (2, 5), "projection_y_coordinate": 1})
    assert w.shape == (3, 7)
    assert w[0, 0] == 1000 + 200 + 10


def test_read_is_cached_and_readonly(cartesian):
    src = Source(cartesian)
    a = src.read("theta", {"time": 0, "full_levels": 0})
    b = src.read("theta", {"time": 0, "full_levels": 0})
    assert a is b
    with pytest.raises(ValueError):
        a[0, 0] = 1


def test_time_labels_and_lead(cartesian):
    src = Source(cartesian)
    t = src.dims["time"]
    assert t.label(1, with_lead=False) == "2000-01-01 00:30"
    assert t.label(1) == "2000-01-01 00:30 (T+1 h 00 min)"  # same fields as T+1 h 30 min


def test_360_day_calendar(latlon):
    src = Source(latlon)
    t = src.dims["time"]
    # day 30 in a 360-day calendar is 1 February
    assert t.label(2) == "2000-01-21"
    assert src.dims["latitude"].is_lat and src.dims["longitude"].is_lon


def test_fill_values_become_nan(latlon):
    src = Source(latlon)
    a = src.read("air_temperature", {"time": 0, "level_height": 0})
    assert np.isnan(a[0, 0])
    assert np.isfinite(a[1, 1])


def test_pressure_levels_point_down(tmp_path):
    src = Source(make_pressure(tmp_path / "p.nc"))
    assert src.dims["plev"].role == Z
    assert src.dims["plev"].positive == "down"


def test_join_files_sorted_by_time(split_files):
    src = Source(split_files)
    assert src.dims["time"].size == 6
    assert [p.split("/")[-1] for p in src.paths] == ["a.nc", "b.nc", "c.nc"]
    for t in range(6):
        assert src.read("theta", {"time": t, "full_levels": 0})[0, 0] == 1000 * t
    series = src.read("theta", {"full_levels": 1, "projection_y_coordinate": 0, "projection_x_coordinate": 0})
    np.testing.assert_array_equal(series, 1000 * np.arange(6) + 100)
    part = src.read("theta", {"time": (1, 5), "full_levels": 0, "projection_y_coordinate": 0,
                              "projection_x_coordinate": 0})
    np.testing.assert_array_equal(part, [1000, 2000, 3000, 4000])
    assert src.dims["time"].label(5, with_lead=False) == "2000-01-01 02:30"


def test_join_rejects_different_grids(tmp_path):
    a = make_cartesian(tmp_path / "a.nc", nx=7)
    b = make_cartesian(tmp_path / "b.nc", nx=8, t0=7200.0)
    with pytest.raises(SourceError, match="projection_x_coordinate"):
        Source([a, b])


def test_missing_file():
    with pytest.raises(SourceError, match="no such file"):
        Source("/nonexistent/file.nc")


def test_nearest_and_find(cartesian):
    src = Source(cartesian)
    half = src.dims["half_levels"]
    assert half.nearest(3.0) in (2, 3)  # 2.5 and 3.5 are equally near
    assert half.nearest(3.2) == 3
    assert half.find("#2") == 2
    assert src.dims["time"].find("2000-01-01 01:00") == 2


def test_value_range(cartesian):
    src = Source(cartesian)
    seen = []
    assert src.value_range("theta", progress=seen.append) == (0.0, 3000 + 500 + 40 + 6)
    assert seen[-1] == 1.0


def test_value_range_along_one_dimension(cartesian):
    """Every time step of level 2, x 1–3, all y: 1000*t + 100*z + 10*y + x."""
    src = Source(cartesian)
    index = {"full_levels": 2, "projection_y_coordinate": (0, 5), "projection_x_coordinate": (1, 4)}
    assert src.value_range("theta", index, "time") == (201.0, 3000 + 200 + 40 + 3)


def test_time_ticks_sit_on_round_time_steps(cartesian):
    """Half-hourly steps from 00:00: four ticks at most → every hour, on the hour."""
    src = Source(cartesian)
    t = src.dims["time"]
    ticks = t.time_ticks(t.values[0], t.values[-1], 2)
    assert ticks == [t.values[0], t.values[2]]
    assert [t.label(t.nearest(v)) .split(" (")[0] for v in ticks] == ["2000-01-01 00:00", "2000-01-01 01:00"]


def test_format_helpers():
    assert format_number(120.5) == "120.5"
    assert format_number(1.25e-05) == "1.25e-05"
    assert format_number(640000.0) == "6.4e+05"
    assert format_lead(1800) == "T+30 min"
    assert format_lead(5400) == "T+1 h 30 min"
    assert format_lead(2 * 86400 + 3600) == "T+2 d 1 h"


def test_lead_times_of_one_axis_share_their_fields():
    """So an animation's title doesn't change length from frame to frame."""
    day = 86400
    assert format_leads([7 * day + 9000, 7 * day + 10800]) == ["T+7 d 02 h 30 min", "T+7 d 03 h 00 min"]
    assert format_leads([0, 1800, 3600]) == ["T+0 h 00 min", "T+0 h 30 min", "T+1 h 00 min"]
    assert format_leads([0, 21600, day]) == ["T+0 d 00 h", "T+0 d 06 h", "T+1 d 00 h"]
    assert format_leads([0.0, float("nan")]) == ["T+0", "T+?"]


@pytest.mark.skipif(not (EXAMPLE_DATA / "lfric_diag_single.nc").exists(), reason="example data not present")
def test_example_lfric_single():
    src = Source(EXAMPLE_DATA / "lfric_diag_single.nc")
    assert src.dims["full_levels"].role == Z
    a = src.read("theta", {"time": 0, "full_levels": 10})
    assert a.shape == (128, 128)
    assert np.isfinite(a).all()


@pytest.mark.skipif(not (EXAMPLE_DATA / "LFRic-Case3-GJ1214b.nc").exists(), reason="example data not present")
def test_example_latlon():
    src = Source(EXAMPLE_DATA / "LFRic-Case3-GJ1214b.nc")
    assert src.dims["latitude"].is_lat
    assert src.dims["longitude"].is_lon
    assert src.dims["level_height"].role == Z


def test_centring_wraps_longitude_round(latlon):
    """Longitude 15..345 every 30°: centred on 0 it runs -165..165, and reads follow."""
    src = Source(latlon)
    before = np.array(src.read("air_temperature", {"time": 1, "level_height": 2}))
    assert src.period("longitude") == 360.0
    src.set_centre("longitude", 0.0)
    lon = src.dims["longitude"]
    assert lon.values[0] == -165.0 and lon.values[-1] == 165.0
    after = src.read("air_temperature", {"time": 1, "level_height": 2})
    np.testing.assert_array_equal(after, np.roll(before, -src.roll("longitude"), axis=1))
    # a window across the old 0/360 seam comes back in one piece
    window = src.read("air_temperature", {"time": 1, "level_height": 2, "longitude": (4, 8)})
    np.testing.assert_array_equal(window, after[:, 4:8])
    assert lon.nearest(350.0) == lon.nearest(-10.0)
    src.set_centre("longitude", None)
    np.testing.assert_array_equal(src.read("air_temperature", {"time": 1, "level_height": 2}), before)


def test_latitude_wraps_from_pole_to_pole(latlon):
    """Latitude -80..80 every 20°. Centred on 90 it wraps like longitude: 0 … 80 in the
    north, then on into the south (-80 … -20), with the poles meeting in the middle."""
    src = Source(latlon)
    before = np.array(src.read("air_temperature", {"time": 1, "level_height": 2}))
    assert src.period("latitude") == 180.0
    src.set_centre("latitude", 90.0)
    lat = src.dims["latitude"]
    np.testing.assert_array_equal(lat.values, [0, 20, 40, 60, 80, 100, 120, 140, 160])
    assert [lat.label(i) for i in range(9)] == ["0", "20", "40", "60", "80", "-80", "-60", "-40", "-20"]
    assert lat.tick_label(90.0) == "±90" and lat.tick_label(120.0) == "-60"
    after = src.read("air_temperature", {"time": 1, "level_height": 2})
    np.testing.assert_array_equal(after, np.roll(before, -4, axis=0))  # same longitudes on every row
    window = src.read("air_temperature", {"time": 1, "level_height": 2, "latitude": (3, 7), "longitude": (2, 5)})
    np.testing.assert_array_equal(window, after[3:7, 2:5])
    assert lat.nearest(-60.0) == 6
    # longitude centred as well
    src.set_centre("longitude", 0.0)
    np.testing.assert_array_equal(src.read("air_temperature", {"time": 1, "level_height": 2}),
                                  np.roll(after, -src.roll("longitude"), axis=1))
    src.set_centre("latitude", None)
    src.set_centre("longitude", None)
    np.testing.assert_array_equal(src.read("air_temperature", {"time": 1, "level_height": 2}), before)


def test_latitude_only_wraps_from_pole_to_pole(tmp_path):
    src = Source(make_pressure(tmp_path / "p.nc"))  # latitude -60..60
    assert src.period("lat") is None
    with pytest.raises(ValueError):
        src.set_centre("lat", 90.0)


def test_latitude_with_points_on_the_poles_labels_each_pole_row():
    """A 90..-90 grid has a north and a south pole row; centred on 90 they sit side by side."""
    from nctui.data import _pole_to_pole

    assert _pole_to_pole(np.arange(90.0, -90.5, -1.0))
    assert not _pole_to_pole(np.arange(-60.0, 90.5, 1.0))


def test_cartesian_x_and_y_count_as_periodic(cartesian):
    src = Source(cartesian)
    assert src.period("projection_x_coordinate") == 700.0
    assert src.period("full_levels") is None
