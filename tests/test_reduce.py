import numpy as np
import pytest

from nctui.reduce import (
    AxisMap,
    bin_edges,
    centred_window,
    every_point_lengths,
    fit_length,
    output_shape,
    reduce_axis,
    resample,
)


def test_bin_edges_cover_everything_without_empty_bins():
    for n in range(1, 60):
        for m in range(1, n + 1):
            e = bin_edges(n, m)
            assert e[0] == 0 and e[-1] == n
            assert (np.diff(e) >= 1).all()


@pytest.mark.parametrize(
    "method, expected",
    [
        ("mean", [1.5, 5.0]),
        ("max", [3.0, 8.0]),
        ("min", [0.0, 2.0]),
        ("nearest", [1.0, 5.0]),
    ],
)
def test_reduce_methods(method, expected):
    a = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 5.0, 8.0, 5.0])
    np.testing.assert_allclose(reduce_axis(a, 0, 2, method), expected)


def test_maxabs_keeps_sign():
    a = np.array([1.0, -9.0, 2.0, 3.0, 4.0, -1.0])
    np.testing.assert_allclose(reduce_axis(a, 0, 2, "maxabs"), [-9.0, 4.0])


def test_non_integer_factor():
    a = np.arange(301, dtype=float)
    out = reduce_axis(a, 0, 104, "mean")
    assert out.shape == (104,)
    assert out[0] < out[-1]


def test_one_level_spike_survives_max_but_not_mean():
    a = np.zeros(301)
    a[150] = 1.0
    assert reduce_axis(a, 0, 100, "max").max() == 1.0
    assert reduce_axis(a, 0, 100, "mean").max() < 0.5


def test_nan_handling():
    a = np.array([np.nan, np.nan, 1.0, np.nan])
    np.testing.assert_array_equal(np.isnan(reduce_axis(a, 0, 2, "mean")), [True, False])
    np.testing.assert_array_equal(np.isnan(reduce_axis(a, 0, 2, "max")), [True, False])


def test_fit_length_uses_whole_repeats():
    assert fit_length(3, 100) == 99
    assert fit_length(128, 151) == 128
    assert fit_length(301, 104) == 104


def test_output_shape_fill_and_equal():
    # z-x section fills the pane
    assert output_shape(128, 301, 151, 104, equal=False) == (128, 104)
    # square map, equal aspect, limited by height
    assert output_shape(128, 128, 181, 104, equal=True, aspect=1.0) == (104, 104)
    # 3x3 grid gets big even blocks
    assert output_shape(3, 3, 181, 104, equal=True, aspect=1.0) == (102, 102)
    # lat/lon 144 x 90 spanning 360 x 180 degrees
    w, h = output_shape(144, 90, 181, 104, equal=True, aspect=2.0)
    assert (w, h) == (144, 90)


def test_every_point_lengths():
    assert every_point_lengths(128, 301, 151, 104, equal=False) == (128, 104)
    assert every_point_lengths(128, 128, 181, 104, equal=True, point_aspect=1.0) == (128, 104)
    # cells twice as wide as tall: each x point needs 2 pixels
    assert every_point_lengths(400, 50, 181, 104, equal=True, point_aspect=2.0) == (90, 50)


def test_every_point_window_really_shows_every_point():
    wx, wy = every_point_lengths(128, 301, 151, 104, equal=False)
    w, h = output_shape(wx, wy, 151, 104, equal=False)
    assert w >= wx and h >= wy


def test_centred_window_stays_inside():
    assert centred_window(301, 104, 0) == (0, 104)
    assert centred_window(301, 104, 300) == (197, 301)
    assert centred_window(301, 104, 150) == (98, 202)
    assert centred_window(10, 50, 5) == (0, 10)


def test_axis_map_round_trip_downsampled():
    m = AxisMap(start=0, n=301, out=104)
    for i in [0, 1, 150, 300]:
        p = m.index_to_pixel(i)
        lo, hi = m.pixel_span(i)
        assert lo == hi == p
        # the pixel's representative index is in the same bin
        assert abs(m.pixel_to_index(p) - i) <= 3


def test_axis_map_upscaled_and_reversed():
    m = AxisMap(start=10, n=3, out=99, reverse=True)
    assert m.pixel_to_index(0) == 12
    assert m.pixel_to_index(98) == 10
    assert m.pixel_span(12) == (0, 32)
    assert m.index_to_pixel(9) is None


def test_resample_orientation():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])  # [y, x]
    out = resample(a, AxisMap(0, 2, 2), AxisMap(0, 2, 2, reverse=True))
    # y increases upwards: last row of data is the top of the screen
    np.testing.assert_array_equal(out, [[3.0, 4.0], [1.0, 2.0]])
