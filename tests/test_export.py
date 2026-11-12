import numpy as np
import pytest

from nctui import export
from nctui.data import Source
from nctui.state import Viewer


@pytest.fixture
def viewer(cartesian):
    v = Viewer(Source(cartesian))
    v.select_var("theta")
    return v


def _png_size(path):
    import matplotlib.image as mpimg

    return mpimg.imread(path).shape[:2]


def test_save_map_png(viewer, tmp_path):
    out = tmp_path / "map.png"
    path = export.save_figure(viewer.src, viewer.request(), str(out), "theta", (4, 3), 50)
    assert out.exists()
    assert path == str(out)
    assert _png_size(out) == (150, 200)


def test_save_uses_given_range_and_zoom(viewer, tmp_path):
    viewer.zoom(2)
    out = tmp_path / "zoom.png"
    export.save_figure(viewer.src, viewer.request(), str(out), "", (4, 3), 50, vrange=(0.0, 5000.0))
    assert out.exists()


def test_save_profile_and_pdf(viewer, tmp_path):
    viewer.open_line("full_levels")
    out = tmp_path / "profile.pdf"
    export.save_figure(viewer.src, viewer.request(), str(out), "profile", (4, 3), 50)
    assert out.read_bytes()[:4] == b"%PDF"


def test_save_latlon_with_coastlines(latlon, tmp_path):
    v = Viewer(Source(latlon))
    v.select_var("air_temperature")
    v.options.coastlines = True
    out = tmp_path / "ll.png"
    export.save_figure(v.src, v.request(), str(out), "", (4, 3), 50)
    assert out.exists()


def test_save_to_missing_folder(viewer, tmp_path):
    with pytest.raises(OSError, match="doesn't exist"):
        export.save_figure(viewer.src, viewer.request(), str(tmp_path / "nope" / "x.png"))


def test_default_filename_and_title(viewer):
    viewer.set_pos("time", 1)
    viewer.set_pos("full_levels", 2)
    req = viewer.request()
    name = export.default_filename(viewer.src, req)
    assert name == "./theta_2000-01-01T0030_full_levels2.png"
    title = export.default_title(viewer.src, req, "map y–x · time …")
    assert title.startswith("air_potential_temperature (K)\n")


def test_edges():
    np.testing.assert_allclose(export._edges(np.array([0.0, 1.0, 2.0])), [-0.5, 0.5, 1.5, 2.5])


def _gif_frames(path):
    from PIL import Image, ImageSequence

    with Image.open(path) as im:
        return [frame.convert("RGB").copy() for frame in ImageSequence.Iterator(im)]


def test_save_animation_gif_one_frame_per_step(viewer, tmp_path):
    seen = []
    out = export.save_animation(viewer.src, viewer.request(), "time", [0, 1, 2, 3], str(tmp_path / "a.gif"),
                                "theta", (3, 2), 40, 4, (0.0, 3546.0), progress=lambda k, n: seen.append((k, n)))
    frames = _gif_frames(out)
    assert len(frames) == 4
    assert seen[-1] == (4, 4)
    assert frames[0].size == (120, 80)
    assert frames[0].tobytes() != frames[3].tobytes()  # the data moved on


def test_save_animation_of_a_profile(viewer, tmp_path):
    viewer.open_line("full_levels")
    out = export.save_animation(viewer.src, viewer.request(), "time", [3, 2, 1], str(tmp_path / "p.gif"),
                                "", (3, 3), 40, 2)
    assert len(_gif_frames(out)) == 3


def test_save_animation_rejects_unknown_formats_and_cancels_cleanly(viewer, tmp_path):
    with pytest.raises(ValueError):
        export.save_animation(viewer.src, viewer.request(), "time", [0, 1], str(tmp_path / "a.avi2"))
    out = tmp_path / "c.gif"
    with pytest.raises(export.Cancelled):
        export.save_animation(viewer.src, viewer.request(), "time", [0, 1, 2], str(out), cancelled=lambda: True)
    assert not out.exists()


@pytest.mark.skipif(not export.can_write_video(), reason="ffmpeg not installed")
def test_save_animation_mp4(viewer, tmp_path):
    out = export.save_animation(viewer.src, viewer.request(), "time", [0, 1, 2, 3], str(tmp_path / "a.mp4"),
                                "", (3, 2), 50, 5)
    with open(out, "rb") as f:
        assert b"ftyp" in f.read(64)


def test_default_animation_filename_leaves_out_the_animated_dimension(viewer):
    name = export.default_animation_filename(viewer.src, viewer.request(), "time")
    assert name.startswith("./theta_full_levels0_time_anim.")


def test_save_map_with_latitude_wrapped(latlon, tmp_path):
    src = Source(latlon)
    v = Viewer(src)
    v.select_var("air_temperature")
    v.centre_on("latitude", 90.0)
    v.options.coastlines = True
    out = export.save_figure(src, v.request(), str(tmp_path / "pole.png"), "pole")
    assert (tmp_path / "pole.png").stat().st_size > 0 and out.endswith("pole.png")


# --------------------------------------------------------------------------- reduced views


def _field(path, name):
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        return np.ma.filled(ds.variables[name][:].astype(np.float64), np.nan)


def _drawn(monkeypatch):
    """Record the arrays handed to matplotlib."""
    drawn = []
    draw_map, draw_line = export._draw_map, export._draw_line

    def spy_map(fig, ax, src, req, data, *args):
        drawn.append(np.array(data))
        return draw_map(fig, ax, src, req, data, *args)

    def spy_line(fig, ax, src, req, *args):
        drawn.append(export._line_values(src, req)[1])
        return draw_line(fig, ax, src, req, *args)

    monkeypatch.setattr(export, "_draw_map", spy_map)
    monkeypatch.setattr(export, "_draw_line", spy_line)
    return drawn


def test_saved_map_is_reduced_from_every_data_point(tmp_path, monkeypatch):
    from .conftest import make_cartesian

    path = make_cartesian(tmp_path / "wide.nc", nt=3, nz=2, ny=40, nx=300)  # wider than any terminal plot
    theta = _field(path, "theta")
    v = Viewer(Source(path))
    v.select_var("theta")
    v.set_pos("full_levels", 1)
    v.set_reduction("time", "max")
    drawn = _drawn(monkeypatch)
    export.save_figure(v.src, v.request(), str(tmp_path / "max.png"), "", (4, 3), 50)
    assert drawn[0].shape == (40, 300)
    np.testing.assert_allclose(drawn[0], theta[:, 1].max(axis=0))
    # zoomed: the window of the same field
    v.set_window("projection_x_coordinate", 100, 220)
    export.save_figure(v.src, v.request(), str(tmp_path / "zoom.png"), "", (4, 3), 50)
    np.testing.assert_allclose(drawn[1], theta[:, 1].max(axis=0)[:, 100:220])


def test_saved_domain_mean_series_and_single_value(latlon, tmp_path, monkeypatch):
    ta = _field(latlon, "air_temperature")
    v = Viewer(Source(latlon))
    v.select_var("air_temperature")
    v.set_pos("level_height", 1)
    v.set_reduction("latitude", "mean")
    v.set_reduction("longitude", "mean")
    v.set_axes("time", None)
    drawn = _drawn(monkeypatch)
    out = tmp_path / "series.png"
    export.save_figure(v.src, v.request(), str(out), "series", (4, 3), 50)
    w = np.cos(np.deg2rad(np.linspace(-80, 80, 9)))[None, :, None] * np.isfinite(ta[:, 1])
    np.testing.assert_allclose(drawn[0], np.nansum(ta[:, 1] * w, axis=(1, 2)) / w.sum(axis=(1, 2)))
    assert out.stat().st_size > 0
    v.set_reduction("time", "mean")
    v.set_reduction("level_height", "mean")
    assert v.kind == "scalar"
    export.save_figure(v.src, v.request(), str(tmp_path / "one.png"), "", (4, 3), 50)
    assert (tmp_path / "one.png").stat().st_size > 0


def test_file_name_and_title_say_what_was_reduced(latlon, viewer):
    viewer.set_pos("full_levels", 2)
    viewer.set_reduction("time", "mean")
    req = viewer.request()
    assert export.default_filename(viewer.src, req) == "./theta_timemean_full_levels2.png"
    assert export.default_animation_filename(viewer.src, req, "full_levels").startswith(
        "./theta_timemean_full_levels_anim.")
    assert export.default_title(viewer.src, req, "map y–x · time mean") == \
        "air_potential_temperature (K)\nmap y–x · time mean"
    v = Viewer(Source(latlon))
    v.select_var("toa_flux")
    v.set_reduction("latitude", "mean")
    v.set_reduction("longitude", "min")
    req = v.request()
    assert export.default_filename(v.src, req) == "./toa_flux_series_latitudemean_longitudemin.png"
    assert export.default_title(v.src, req, "") == "toa_flux (W m-2), cos(lat)-weighted mean"


def test_animation_of_a_reduced_view(viewer, tmp_path, monkeypatch):
    theta = _field(viewer.src.paths[0], "theta")
    viewer.set_reduction("time", "mean")
    titles = []
    drawn = _drawn(monkeypatch)
    monkeypatch.setattr(export, "_fit_title", lambda text, width, fontsize=10: titles.append(text) or text)
    out = export.save_animation(viewer.src, viewer.request(), "full_levels", [0, 2, 5], str(tmp_path / "m.gif"),
                                "theta", (3, 2), 40, 4)
    assert len(_gif_frames(out)) == 3
    np.testing.assert_allclose(drawn[0], theta[:, 0].mean(axis=0))
    assert titles[-1] == "theta\nmap y–x · time mean · full_levels 5"
    assert viewer.src.reduce_cached("theta", {"full_levels": 5}, (("time", "mean"),))


def test_animation_of_a_reduced_view_can_be_cancelled_while_reducing(viewer, tmp_path, monkeypatch):
    from nctui import data

    monkeypatch.setattr(data, "REDUCE_BLOCK_BYTES", 1)  # several blocks, so there is a "part way"
    viewer.set_reduction("time", "mean")
    out = tmp_path / "c.gif"
    asked = []
    with pytest.raises(export.Cancelled):
        export.save_animation(viewer.src, viewer.request(), "full_levels", [0, 1, 2], str(out),
                              cancelled=lambda: asked.append(1) or len(asked) > 2)
    assert not out.exists()
    assert not viewer.src.reduce_cached("theta", {"full_levels": 0}, (("time", "mean"),))
