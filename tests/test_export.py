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
