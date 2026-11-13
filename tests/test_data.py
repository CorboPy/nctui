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


def test_latitude_is_mirrored_at_the_poles(latlon):
    """Latitude -80..80 every 20°. Centred on 90: from the south pole up to the north
    pole in the middle, then mirrored back down to the south pole, same longitudes."""
    src = Source(latlon)
    before = np.array(src.read("air_temperature", {"time": 1, "level_height": 2}))
    assert src.period("latitude") == 360.0
    src.set_centre("latitude", 90.0)
    lat = src.dims["latitude"]
    assert lat.size == 18  # every row twice
    np.testing.assert_array_equal(lat.values, np.arange(-80, 261, 20))
    assert [lat.label(i) for i in (0, 8, 9, 17)] == ["-80", "80", "80", "-80"]
    assert lat.tick_label(90.0) == "90" and lat.tick_label(120.0) == "60"
    after = src.read("air_temperature", {"time": 1, "level_height": 2})
    np.testing.assert_array_equal(after, np.concatenate([before, before[::-1]]))  # a mirror, same longitudes
    window = src.read("air_temperature", {"time": 1, "level_height": 2, "latitude": (6, 12), "longitude": (2, 5)})
    np.testing.assert_array_equal(window, after[6:12, 2:5])
    np.testing.assert_array_equal(src.read("air_temperature", {"time": 1, "level_height": 2, "latitude": 13}),
                                  after[13])
    # centred on -90 the mirror is about the south pole: north pole at the bottom and the top
    src.set_centre("latitude", -90.0)
    np.testing.assert_array_equal(src.dims["latitude"].values, np.arange(-260, 81, 20))
    np.testing.assert_array_equal(src.read("air_temperature", {"time": 1, "level_height": 2}),
                                  np.concatenate([before[::-1], before]))
    # longitude centred as well
    src.set_centre("longitude", 0.0)
    np.testing.assert_array_equal(src.read("air_temperature", {"time": 1, "level_height": 2}),
                                  np.roll(np.concatenate([before[::-1], before]), -src.roll("longitude"), axis=1))
    with pytest.raises(ValueError):
        src.set_centre("latitude", 100.0)
    src.set_centre("latitude", None)
    src.set_centre("longitude", None)
    assert src.dims["latitude"].size == 9
    np.testing.assert_array_equal(src.read("air_temperature", {"time": 1, "level_height": 2}), before)


def test_latitude_only_wraps_from_pole_to_pole(tmp_path):
    src = Source(make_pressure(tmp_path / "p.nc"))  # latitude -60..60
    assert src.period("lat") is None
    with pytest.raises(ValueError):
        src.set_centre("lat", 90.0)


def test_latitude_with_points_on_the_poles_labels_each_pole_row():
    """A grid that reaches both poles can be centred; one that stops short can't."""
    from nctui.data import _pole_to_pole

    assert _pole_to_pole(np.arange(90.0, -90.5, -1.0))
    assert not _pole_to_pole(np.arange(-60.0, 90.5, 1.0))


def test_cartesian_x_and_y_count_as_periodic(cartesian):
    src = Source(cartesian)
    assert src.period("projection_x_coordinate") == 700.0
    assert src.period("full_levels") is None


def test_mirror_layout_keeps_one_copy_of_a_row_on_the_pole():
    from nctui.data import _mirror_layout, fold_latitude

    u, rows = _mirror_layout(np.arange(90.0, -90.5, -30.0), 90.0)  # 90 … -90, descending
    assert len(u) == 12 and list(u).count(90.0) == 1 and list(u).count(-90.0) == 1
    assert np.all(np.diff(u) == -30.0)
    assert fold_latitude(100.0) == 80.0 and fold_latitude(-100.0) == -80.0 and fold_latitude(45.0) == 45.0


# --------------------------------------------------------------------------- reducing over dimensions


def _values(path, name):
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        return np.ma.filled(ds.variables[name][:].astype(np.float64), np.nan)


def _nan(fn, a, axis):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        return fn(a, axis=axis)


@pytest.mark.parametrize("stat, fn", [("mean", np.mean), ("min", np.min), ("max", np.max), ("sum", np.sum)])
def test_reduce_over_time_matches_numpy(cartesian, stat, fn):
    src = Source(cartesian)
    theta = _values(cartesian, "theta")
    out = src.reduce_read("theta", {"full_levels": 2}, (("time", stat),))
    assert out.shape == (5, 7) and out.dtype == np.float64
    np.testing.assert_allclose(out, fn(theta[:, 2], axis=0))


@pytest.mark.parametrize("stat, fn", [("mean", np.nanmean), ("min", np.nanmin), ("max", np.nanmax)])
def test_reduce_ignores_missing_values(gappy, stat, fn):
    src = Source(gappy)
    sst = _values(gappy, "sst")
    out = src.reduce_read("sst", {}, (("time", stat),))
    np.testing.assert_allclose(out, _nan(fn, sst, 0))
    assert np.isnan(out[1, 2]) and np.isfinite(out).sum() == out.size - 1  # only the all-missing column


def test_sum_of_nothing_is_missing_not_zero(gappy):
    src = Source(gappy)
    sst = _values(gappy, "sst")
    out = src.reduce_read("sst", {}, (("time", "sum"),))
    expected = np.nansum(sst, axis=0)
    expected[1, 2] = np.nan
    np.testing.assert_allclose(out, expected)


def test_mean_over_latitude_is_cos_weighted(gappy):
    src = Source(gappy)
    sst = _values(gappy, "sst")
    w = np.cos(np.deg2rad(np.linspace(-60, 60, 4)))[None, :, None] * np.isfinite(sst)
    with np.errstate(invalid="ignore"):
        expected = np.nansum(sst * w, axis=1) / w.sum(axis=1)  # 0/0 where a column is all missing
    out = src.reduce_read("sst", {}, (("lat", "mean"),))
    np.testing.assert_allclose(out, expected)
    assert not np.allclose(out, np.nanmean(sst, axis=1), equal_nan=True)  # the plain mean differs
    assert src.reduce_weighting("sst", (("lat", "mean"),)) == "cos(lat)-weighted mean"
    # other statistics, and a mean that leaves latitude alone, are not weighted
    np.testing.assert_allclose(src.reduce_read("sst", {}, (("lat", "max"),)), _nan(np.nanmax, sst, 1))
    np.testing.assert_allclose(src.reduce_read("sst", {}, (("lon", "mean"),)), _nan(np.nanmean, sst, 2))
    assert src.reduce_weighting("sst", (("lon", "mean"),)) == "unweighted mean"
    assert src.reduce_weighting("sst", (("lat", "max"),)) == ""


def test_mean_over_a_cartesian_y_is_not_weighted(cartesian):
    src = Source(cartesian)
    theta = _values(cartesian, "theta")
    out = src.reduce_read("theta", {"time": 1, "full_levels": 0}, (("projection_y_coordinate", "mean"),))
    np.testing.assert_allclose(out, theta[1, 0].mean(axis=0))
    assert src.reduce_weighting("theta", (("projection_y_coordinate", "mean"),)) == "unweighted mean"


def test_dimensions_sharing_a_statistic_collapse_together(gappy):
    """A domain mean is one weighted mean of every valid point, not a mean of zonal means."""
    src = Source(gappy)
    sst = _values(gappy, "sst")
    w = np.cos(np.deg2rad(np.linspace(-60, 60, 4)))[None, :, None] * np.isfinite(sst)
    series = src.reduce_read("sst", {}, (("lat", "mean"), ("lon", "mean")))
    np.testing.assert_allclose(series, np.nansum(sst * w, axis=(1, 2)) / w.sum(axis=(1, 2)))
    everything = src.reduce_read("sst", {}, (("time", "mean"), ("lat", "mean"), ("lon", "mean")))
    assert everything.shape == ()
    np.testing.assert_allclose(everything, np.nansum(sst * w) / w.sum())


def test_different_statistics_apply_from_the_last_dimension_to_the_first(gappy):
    src = Source(gappy)
    sst = _values(gappy, "sst")
    out = src.reduce_read("sst", {}, (("time", "max"), ("lon", "mean")))
    np.testing.assert_allclose(out, _nan(np.nanmax, _nan(np.nanmean, sst, 2), 0))


def test_reduce_reads_in_blocks_with_progress_and_cancel(gappy, monkeypatch):
    from nctui import data

    src = Source(gappy)
    sst = _values(gappy, "sst")
    whole = src.reduce_read("sst", {}, (("time", "mean"), ("lat", "mean")))
    monkeypatch.setattr(data, "REDUCE_BLOCK_BYTES", 8 * 6 * 2)  # two latitudes of one time step at once
    src = Source(gappy)
    seen = []
    out = src.reduce_read("sst", {}, (("time", "mean"), ("lat", "mean")), progress=seen.append)
    np.testing.assert_allclose(out, whole)
    assert len(seen) == 10 and seen[-1] == 1.0 and seen == sorted(seen)
    # cancelling part way gives nothing, and leaves nothing half-done behind
    src = Source(gappy)
    calls = []
    assert src.reduce_read("sst", {}, (("time", "max"),), cancelled=lambda: calls.append(1) or len(calls) > 2) is None
    assert not src.reduce_cached("sst", {}, (("time", "max"),))
    np.testing.assert_allclose(src.reduce_read("sst", {}, (("time", "max"),)), _nan(np.nanmax, sst, 0))


def test_reduced_result_is_cached_whatever_the_window(cartesian, monkeypatch):
    src = Source(cartesian)
    theta = _values(cartesian, "theta")
    reduce = (("time", "mean"),)
    assert not src.reduce_cached("theta", {"full_levels": 1}, reduce)
    full = src.reduce_read("theta", {"full_levels": 1}, reduce)
    assert src.reduce_cached("theta", {"full_levels": 1, "projection_x_coordinate": (2, 5)}, reduce)
    assert not src.reduce_cached("theta", {"full_levels": 2}, reduce)  # another level is another field
    monkeypatch.setattr(src, "_slice", lambda *a: pytest.fail("read the file again"))
    zoomed = src.reduce_read("theta", {"full_levels": 1, "projection_x_coordinate": (2, 5),
                                       "projection_y_coordinate": (1, 3)}, reduce)
    np.testing.assert_allclose(zoomed, theta[:, 1].mean(axis=0)[1:3, 2:5])
    np.testing.assert_allclose(zoomed, full[1:3, 2:5])
    with pytest.raises(ValueError):
        full[0, 0] = 1  # read-only, like every read


def test_reduce_over_joined_files(split_files):
    src = Source(split_files)
    parts = [_values(p, "theta") for p in sorted(split_files, key=lambda p: p.name)]  # a, b, c: in time order
    theta = np.concatenate(parts, axis=0)
    assert theta.shape[0] == 6
    out = src.reduce_read("theta", {"full_levels": 0}, (("time", "mean"),))
    np.testing.assert_allclose(out, theta[:, 0].mean(axis=0))
    assert out[0, 0] == 2500.0  # 1000 × the mean of time steps 0…5
    series = src.reduce_read("theta", {"full_levels": 3}, (("projection_y_coordinate", "max"),
                                                           ("projection_x_coordinate", "max")))
    np.testing.assert_allclose(series, theta[:, 3].max(axis=(1, 2)))


def test_reduce_follows_a_centred_longitude_without_reading_again(latlon, monkeypatch):
    src = Source(latlon)
    ta = _values(latlon, "air_temperature")
    reduce = (("time", "mean"),)
    plain = src.reduce_read("air_temperature", {"level_height": 0}, reduce)
    np.testing.assert_allclose(plain, np.nanmean(ta[:, 0], axis=0))
    with monkeypatch.context() as m:
        m.setattr(src, "_slice", lambda *a: pytest.fail("read the file again"))
        src.set_centre("longitude", 0.0)
        shift = src.roll("longitude")
        assert shift
        wrapped = src.reduce_read("air_temperature", {"level_height": 0}, reduce)
        np.testing.assert_allclose(wrapped, np.roll(plain, -shift, axis=1))
        window = src.reduce_read("air_temperature", {"level_height": 0, "longitude": (4, 9)}, reduce)
        np.testing.assert_allclose(window, np.roll(plain, -shift, axis=1)[:, 4:9])
    # a position along the centred axis is the same grid point as before
    j = 5
    line = src.reduce_read("air_temperature", {"level_height": 0, "longitude": j}, (("latitude", "max"),))
    np.testing.assert_allclose(line, np.nanmax(ta[:, 0, :, src.stored_index("longitude", j)], axis=1))


def test_reduce_counts_each_row_of_a_mirrored_latitude_once(latlon):
    ta = _values(latlon, "air_temperature")
    src = Source(latlon)
    src.set_centre("latitude", -90.0)  # centred before anything is reduced: nothing comes from the cache
    assert src.dims["latitude"].size == 18  # every row twice on screen
    # collapsing the mirrored latitude still sums the file's nine rows
    total = src.reduce_read("air_temperature", {"time": 1, "level_height": 2}, (("latitude", "sum"),))
    np.testing.assert_allclose(total, ta[1, 2].sum(axis=0))
    weighted = src.reduce_read("air_temperature", {"time": 1, "level_height": 2}, (("latitude", "mean"),))
    w = np.cos(np.deg2rad(np.linspace(-80, 80, 9)))[:, None]
    np.testing.assert_allclose(weighted, (ta[1, 2] * w).sum(axis=0) / w.sum())
    # a mirrored latitude that is kept shows each row of the time mean twice
    shown = src.reduce_read("air_temperature", {"level_height": 0}, (("time", "mean"),))
    assert shown.shape == (18, 12)
    np.testing.assert_allclose(shown, np.nanmean(ta[:, 0], axis=0)[src._mirror["latitude"]])
    # and a position along it is the file's row, whichever copy the probe is on
    j = 15
    series = src.reduce_read("air_temperature", {"level_height": 0, "latitude": j}, (("longitude", "max"),))
    np.testing.assert_allclose(series, ta[:, 0, src.stored_index("latitude", j)].max(axis=1))
    # undoing the centring gives the same fields back, in the file's order
    src.set_centre("latitude", None)
    np.testing.assert_allclose(src.reduce_read("air_temperature", {"level_height": 0}, (("time", "mean"),)),
                               np.nanmean(ta[:, 0], axis=0))


def test_reduce_rejects_unknown_dimensions_and_statistics(cartesian):
    src = Source(cartesian)
    with pytest.raises(ValueError):
        src.reduce_read("theta", {}, (("half_levels", "mean"),))
    with pytest.raises(ValueError):
        src.reduce_read("theta", {}, (("time", "median"),))


def test_value_range_of_a_reduced_view(cartesian):
    src = Source(cartesian)
    theta = _values(cartesian, "theta")
    seen = []
    rng = src.value_range("theta", {"projection_x_coordinate": (0, 7), "projection_y_coordinate": (0, 5)},
                          "full_levels", seen.append, reduce=(("time", "mean"),))
    means = theta.mean(axis=0)
    assert rng == (means.min(), means.max())
    assert seen[-1] == 1.0 and seen == sorted(seen)
    assert src.reduce_cached("theta", {"full_levels": 4}, (("time", "mean"),))  # an animation reads these next


def test_same_statistic_collapses_together_across_a_kept_dimension(gappy):
    """time and lon are not neighbours in (time, lat, lon), but one mean still covers both."""
    src = Source(gappy)
    sst = _values(gappy, "sst")
    out = src.reduce_read("sst", {}, (("time", "mean"), ("lon", "mean")))
    np.testing.assert_allclose(out, _nan(np.nanmean, sst, (0, 2)))
    mean_of_means = _nan(np.nanmean, _nan(np.nanmean, sst, 2), 0)
    assert not np.allclose(out, mean_of_means, equal_nan=True)  # which is what separate groups would give


@pytest.mark.parametrize("reduce", [(("lon", "mean"),), (("lat", "max"),), (("time", "sum"), ("lon", "sum")),
                                    (("time", "min"),)])
def test_blocks_stay_within_the_budget_when_kept_dimensions_are_large(gappy, monkeypatch, reduce):
    """A zonal mean of a long time × lat field is tiled over time and lat too, not read in one go."""
    from nctui import data

    sst = _values(gappy, "sst")
    whole = Source(gappy).reduce_read("sst", {}, reduce)
    monkeypatch.setattr(data, "REDUCE_BLOCK_BYTES", 8 * 6 * 2)
    src = Source(gappy)
    sizes = []
    read = src._slice

    def spy(*args):
        out = read(*args)
        sizes.append(out.nbytes)
        return out

    monkeypatch.setattr(src, "_slice", spy)
    seen = []
    out = src.reduce_read("sst", {}, reduce, progress=seen.append)
    assert max(sizes) <= 8 * 6 * 2 and len(sizes) > 1
    np.testing.assert_allclose(out, whole)
    assert out.shape == tuple(n for n, d in zip(sst.shape, ("time", "lat", "lon"), strict=True)
                              if d not in dict(reduce))
    assert seen[-1] == 1.0 and seen == sorted(seen)
