"""Drive the real app with Textual's pilot and check what it does."""

import os

import pytest

from nctui.app import NctuiApp
from nctui.data import Source
from nctui.screens.colours import ColoursScreen
from nctui.screens.info import HelpScreen, InfoScreen
from nctui.screens.menu import ActionMenu
from nctui.screens.save import SaveScreen
from nctui.screens.settings import PlotSettingsScreen
from nctui.widgets.plot import PlotView

SIZE = (140, 44)


async def settle(pilot, app, t=0.1):
    await pilot.pause(t)
    for _ in range(50):
        if not app._building and app._pending is None:
            break
        await pilot.pause(0.02)
    await pilot.pause()


async def wait_for(pilot, condition, tries=100):
    for _ in range(tries):
        if condition():
            return True
        await pilot.pause(0.05)
    return condition()


@pytest.fixture
def app(cartesian):
    return NctuiApp(Source(cartesian), "theta")


async def test_opens_and_draws(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        assert app.viewer.var == "theta"
        assert app.query_one(PlotView).frame is not None
        assert "theta" in str(app.query_one("#main").border_title)


async def test_variable_list_plots_as_you_move(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("down")
        await settle(pilot, app, 0.2)
        assert app.viewer.var != "theta"
        assert app.last_frame.request.var == app.viewer.var


async def test_brackets_step_time_and_level(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("right_square_bracket", "right_square_bracket", "right_curly_bracket")
        await settle(pilot, app)
        assert app.viewer.pos("time") == 2
        assert app.viewer.pos("full_levels") == 1
        assert "time 2000-01-01 01:00" in app.last_frame.header


async def test_view_cycle_and_swap(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("v")
        await settle(pilot, app)
        assert app.viewer.ydim == "full_levels"
        await pilot.press("t")
        await settle(pilot, app)
        assert app.viewer.xdim == "full_levels"


async def test_profile_through_probe_and_back(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.query_one(PlotView).focus()
        await pilot.press("right", "right")
        await settle(pilot, app)
        x = app.viewer.pos("projection_x_coordinate")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ActionMenu)
        await pilot.press("enter")  # first option: vertical profile
        await settle(pilot, app)
        assert app.viewer.kind == "line" and app.viewer.xdim == "full_levels"
        assert app.last_frame.line_fig is not None
        await pilot.press("escape")
        await settle(pilot, app)
        assert app.viewer.kind == "map"
        assert app.viewer.pos("projection_x_coordinate") == x


async def test_click_sets_probe(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        f = app.last_frame
        plot = app.query_one(PlotView)
        await pilot.click(PlotView, offset=(f.image_col, f.image_row))
        await settle(pilot, app)
        assert plot.has_focus
        # top-left pixel: first x, last y (y increases upwards)
        assert app.viewer.pos("projection_x_coordinate") == 0
        assert app.viewer.pos("projection_y_coordinate") == 4


async def test_every_point_and_fit(tmp_path):
    from .conftest import make_cartesian

    app = NctuiApp(Source(make_cartesian(tmp_path / "tall.nc", nt=1, nz=301, ny=3, nx=40)), "theta")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("v")
        await settle(pilot, app)
        assert "÷" in app.last_frame.status  # 301 levels squeezed
        await pilot.press("1")
        await settle(pilot, app)
        assert app.viewer.is_zoomed()
        assert "of 301" in app.last_frame.status
        assert "÷" not in app.last_frame.status
        await pilot.press("0")
        await settle(pilot, app)
        assert not app.viewer.is_zoomed()


async def test_colours_popup_applies_live(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, ColoursScreen)
        await pilot.press("down")
        await settle(pilot, app)
        assert app.viewer.colour.cmap == "plasma"
        assert app.last_frame.request.colour.cmap == "plasma"
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ColoursScreen)


async def test_quick_keys(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("m")
        await settle(pilot, app)
        assert app.viewer.colour.cmap == "plasma"
        await pilot.press("r")
        await settle(pilot, app)
        assert app.viewer.colour.mode == "locked"
        await pilot.press("o")  # not a lat/lon map: stays off
        assert not app.viewer.options.coastlines


async def test_settings_and_help_and_info(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        for key, screen in (("p", PlotSettingsScreen), ("question_mark", HelpScreen), ("i", InfoScreen)):
            await pilot.press(key)
            await pilot.pause()
            assert isinstance(app.screen, screen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, screen)


async def test_save_png(app, tmp_path):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("e")
        await pilot.pause()
        assert isinstance(app.screen, SaveScreen)
        out = tmp_path / "out.png"
        app.screen.query_one("#file").value = str(out)
        await pilot.press("enter")
        for _ in range(600):  # up to 30 s: a fresh CI runner can be slow to draw the first figure
            if out.exists() and not isinstance(app.screen, SaveScreen):
                break
            await pilot.pause(0.05)
        assert out.exists() and os.path.getsize(out) > 1000
        assert not isinstance(app.screen, SaveScreen)


async def test_save_animation_while_playing(app, tmp_path):
    """e during playback opens the dialog on animation, at the playback speed."""
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("space")
        await pilot.press("e")
        await pilot.pause()
        assert isinstance(app.screen, SaveScreen) and app.screen.anim
        assert app.screen.query_one("#fps").value == "5"
        assert app.screen.query_one("#frames").display
        out = tmp_path / "anim.gif"
        app.screen.query_one("#file").value = str(out)
        app.screen.query_one("#last").value = "#2"  # three frames: 0, 1, 2
        app.screen.query_one("#dpi").value = "30"
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: out.exists() and not isinstance(app.screen, SaveScreen), 200)
        from PIL import Image

        with Image.open(out) as im:
            assert im.n_frames == 3


async def test_switching_save_dialog_to_animation_changes_the_file_type(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("e")
        await pilot.pause()
        assert not app.screen.anim and not app.screen.query_one("#frames").display
        app.screen.query_one("#what").query("RadioButton")[1].value = True
        await pilot.pause()
        assert app.screen.anim
        assert app.screen.query_one("#file").value.endswith(("_time_anim.mp4", "_time_anim.gif"))


async def test_save_dialog_animation_format_choice(app, monkeypatch):
    import nctui.screens.save as save

    monkeypatch.setattr(save, "can_write_video", lambda: True)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("e")
        await pilot.pause()
        screen = app.screen
        assert not screen.query_one("#format").display  # only for animations
        screen.query_one("#what").query("RadioButton")[1].value = True
        await pilot.pause()
        fmt = screen.query_one("#format")
        file = screen.query_one("#file")
        assert fmt.display and file.value.endswith(".mp4")
        await pilot.click(fmt.query("RadioButton")[1])  # gif
        await pilot.pause()
        assert file.value.endswith("_time_anim.gif")
        file.value = "movie.mp4"  # typing a name picks the format
        await pilot.pause()
        assert fmt.pressed_index == 0
        screen.query_one("#what").query("RadioButton")[0].value = True  # back to a plot
        await pilot.pause()
        assert file.value == "movie.png"


async def test_save_dialog_format_ignores_a_stale_file_name_event(app, monkeypatch):
    """Two file-name changes queued before either is handled: the older one must not
    switch the format. (Without ffmpeg this is what happens when you choose animation:
    the default .gif name is set, then the chosen format's extension replaces it.)"""
    import nctui.screens.save as save

    monkeypatch.setattr(save, "can_write_video", lambda: True)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("e")
        await pilot.pause()
        screen = app.screen
        screen.query_one("#what").query("RadioButton")[1].value = True
        await pilot.pause()
        fmt, file = screen.query_one("#format"), screen.query_one("#file")
        assert fmt.pressed_index == 0  # mp4
        file.value = "movie.gif"  # queued, then overtaken before it is handled
        file.value = "movie.mp4"
        await pilot.pause()
        assert fmt.pressed_index == 0 and file.value == "movie.mp4"
        await pilot.click(fmt.query("RadioButton")[1])  # and choosing gif sticks
        await pilot.pause()
        assert fmt.pressed_index == 1 and file.value == "movie.gif"


async def test_save_dialog_choice_stays_visible_and_works_by_keyboard_and_mouse(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("e")
        await pilot.pause()
        what = app.screen.query_one("#what")
        await pilot.press("shift+tab")  # from the file name back to plot / animation
        await pilot.pause()
        assert what.has_focus and what.content_region.height == 1
        await pilot.press("right", "enter")
        await pilot.pause()
        assert app.screen.anim
        await pilot.click(what.query("RadioButton")[0])
        await pilot.pause()
        assert not app.screen.anim


async def test_centre_the_map_from_the_plot_menu(latlon):
    app = NctuiApp(Source(latlon), "air_temperature")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.query_one(PlotView).focus()
        await pilot.press("enter")
        await pilot.pause()
        ids = [o.id for o in app.screen.query_one("OptionList")._options]
        assert "centre" in ids
        app.screen.dismiss("centre")
        await settle(pilot, app)
        assert app.viewer.src.centre("longitude") is not None
        assert app.viewer.src.centre("latitude") is None  # only over the pole when asked
        assert "centred" in str(app.query_one("DimensionPanel").render())


async def test_centre_with_latitude_from_the_plot_menu(latlon):
    app = NctuiApp(Source(latlon), "air_temperature")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.viewer.set_pos("latitude", 7)  # 60°N
        app.query_one(PlotView).focus()
        await pilot.press("enter")
        await pilot.pause()
        options = {o.id: str(o.prompt) for o in app.screen.query_one("OptionList")._options}
        assert options["centre"] == "Centre here: longitude"
        assert options["centre_lat"] == "Centre here: longitude + latitude"
        app.screen.dismiss("centre_lat")
        await settle(pilot, app)
        assert app.viewer.src.centre("latitude") == 60.0


async def test_save_png_bad_folder_keeps_dialog_open(app, tmp_path):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#file").value = str(tmp_path / "missing" / "x.png")
        await pilot.press("enter")
        await pilot.pause(0.5)
        assert isinstance(app.screen, SaveScreen)
        assert "couldn't save" in str(app.screen.query_one("#message").render())


async def test_search_filters_variables(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("slash")
        await pilot.press("r", "h", "o")
        await pilot.pause()
        tree = app.query_one("#vars")
        assert list(tree._leaves) == ["rho"]
        await pilot.press("enter")
        await settle(pilot, app, 0.2)
        assert app.viewer.var == "rho"
        await pilot.press("escape")
        await pilot.pause()
        assert "theta" in tree._leaves


async def test_animation_advances_time(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("greater_than_sign", "greater_than_sign", "greater_than_sign")  # 20 fps
        await pilot.press("space")
        # watch the steps go by: where it stops says nothing, as 4 steps wrap round to 0
        seen = {app.viewer.pos("time")}
        assert await wait_for(pilot, lambda: seen.add(app.viewer.pos("time")) or len(seen) >= 3, tries=200)
        await pilot.press("space")
        await settle(pilot, app)
        assert app._anim_timer is None


async def test_playback_waits_for_a_slow_terminal_and_says_how_fast_it_really_is(app):
    import time

    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("space")
        await settle(pilot, app)
        app._output_behind = lambda: True  # the terminal hasn't taken the last frame yet
        here = app.viewer.pos("time")
        app._anim_tick()
        assert app.viewer.pos("time") == here  # skipped, not queued
        now = time.monotonic()
        app._anim_started = now - 3
        app._shown.clear()
        app._shown.extend(now - 1.8 + 0.6 * k for k in range(4))  # ~1.7 frames a second shown
        app._update_topbar()
        assert "▶ playing time · 2 of 5 fps" in str(app.query_one("#topbar-left").render())
        await pilot.press("space")


async def test_play_shows_no_tip(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("space")
        await pilot.pause(0.2)
        assert app._anim_timer is not None
        assert not list(app._notifications)
        await pilot.press("space")


async def test_play_with_one_time_step_says_why_in_red(tmp_path):
    from .conftest import make_cartesian

    app = NctuiApp(Source(make_cartesian(tmp_path / "one.nc", nt=1)), "theta")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("space")
        await pilot.pause()
        assert app._anim_timer is None
        bar = app.query_one("#topbar-left").render()
        assert "■ can't play: only 1 time step" in str(bar)
        assert any("red" in str(span.style) for span in bar.spans)


async def test_view_change_while_animating_levels_stops_instead_of_crashing(tmp_path):
    """One time step, so animate the levels; then v puts the levels on the plot."""
    from .conftest import make_cartesian

    app = NctuiApp(Source(make_cartesian(tmp_path / "one.nc", nt=1)), "theta")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app._start_animation("full_levels")
        await pilot.pause(0.3)
        assert app._anim_timer is not None
        await pilot.press("v")
        await settle(pilot, app)
        assert app._anim_timer is None
        assert app.last_frame.error is None and "full_levels" in (app.viewer.xdim, app.viewer.ydim)
        assert "■ stopped: view changed" in str(app.query_one("#topbar-left").render())


async def test_dimension_panel_keys(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("tab")  # Variables → Dimensions
        dims = app.query_one("#dims")
        assert dims.has_focus
        await pilot.press("down", "right", "right")  # full_levels +2
        await settle(pilot, app)
        assert app.viewer.pos("full_levels") == 2
        await pilot.press("y")  # full_levels on the y axis
        await settle(pilot, app)
        assert app.viewer.ydim == "full_levels"


async def test_latlon_coastlines_toggle(latlon):
    app = NctuiApp(Source(latlon), "air_temperature")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("o")
        await settle(pilot, app)
        assert app.viewer.options.coastlines
        assert app.last_frame.request.options.coastlines


async def test_drag_zooms_to_box(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        f = app.last_frame
        x0, y0 = f.image_col, f.image_row
        await pilot.mouse_down(PlotView, offset=(x0 + 1, y0 + 1))
        await pilot.hover(PlotView, offset=(x0 + 25, y0 + 8))
        await pilot.mouse_up(PlotView, offset=(x0 + 25, y0 + 8))
        await settle(pilot, app)
        assert app.viewer.is_zoomed()
        assert app.viewer.selection is None


async def test_goto_popup(app):
    from nctui.screens.goto import GoToScreen

    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("g")
        await pilot.pause()
        assert isinstance(app.screen, GoToScreen)
        app.screen.query_one("#dim-0").value = "2000-01-01 01:00"
        app.screen.query_one("#dim-1").value = "#4"
        app.screen.query_one("#dim-3").value = "100:400"
        app.screen.query_one("#dim-0").focus()
        await pilot.press("enter")
        await settle(pilot, app)
        assert not isinstance(app.screen, GoToScreen)
        assert app.viewer.pos("time") == 2
        assert app.viewer.pos("full_levels") == 4
        assert app.viewer.window("projection_x_coordinate") == (1, 5)


async def test_goto_rejects_bad_value(app):
    from nctui.screens.goto import GoToScreen

    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("g")
        await pilot.pause()
        app.screen.query_one("#dim-1").value = "not a number"
        app.screen.query_one("#dim-1").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, GoToScreen)


async def test_click_on_profile_moves_marker(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.viewer.open_line("full_levels")
        app.view_changed()
        await settle(pilot, app)
        lf = app.last_frame.line_fig
        # click near the top of the profile: the highest level
        await pilot.click(PlotView, offset=(lf.plot_col + 2, lf.plot_row))
        await settle(pilot, app)
        assert app.viewer.pos("full_levels") == 5


async def test_animation_range_scans_only_the_slice_being_animated(app):
    """theta = 1000*t + 100*z + 10*y + x: on level 2, all times span 200 .. 3000+200+46."""
    from textual.widgets import RadioSet

    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("right_curly_bracket", "right_curly_bracket")  # level 2
        await settle(pilot, app)
        await pilot.press("c")
        await pilot.pause()
        radios = app.screen.query_one("#mode", RadioSet)
        assert str(radios.query("RadioButton")[2].label) == "min/max: fit animation"
        radios.query("RadioButton")[2].value = True
        assert await wait_for(pilot, lambda: app.viewer.anim_ranges)
        await settle(pilot, app)
        assert app.viewer.colour.mode == "anim"
        assert app.last_frame.vrange == (200.0, 3246.0)
        assert "range of all time steps" in app.last_frame.stats
        # another level is another slice: it is scanned again
        await pilot.press("escape", "right_curly_bracket")
        assert await wait_for(pilot, lambda: len(app.viewer.anim_ranges) == 2)
        await settle(pilot, app)
        assert app.last_frame.vrange == (300.0, 3346.0)


async def test_manual_range_boxes_appear_for_manual_and_apply_on_tab(app):
    from textual.widgets import Input, RadioSet

    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("c")
        await pilot.pause()
        assert not app.screen.query_one("#manual").display
        app.screen.query_one("#mode", RadioSet).query("RadioButton")[3].value = True
        await pilot.pause()
        assert app.screen.query_one("#manual").display
        vmin = app.screen.query_one("#vmin", Input)
        assert vmin.has_focus
        vmin.value = "-5"
        await pilot.press("tab")  # leaving the box applies it, no enter needed
        await settle(pilot, app)
        assert app.viewer.colour.mode == "manual"
        assert app.last_frame.vrange[0] == -5.0
