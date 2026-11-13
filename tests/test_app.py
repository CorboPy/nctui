"""Drive the real app with Textual's pilot and check what it does."""

import os

import pytest

from nctui.app import NctuiApp
from nctui.data import Source
from nctui.screens.colours import ColoursScreen
from nctui.screens.info import HelpScreen, InfoScreen
from nctui.screens.menu import ActionMenu, Entry
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


def _options(app):
    return {e.id: e.label for e in app.screen.entries.values()}


async def test_menus_share_sections_and_show_keys(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.query_one(PlotView).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert [name for name, _ in app.screen.sections] == ["Here", "Reduce", "View", "Plot"]
        assert not any(k.startswith("reduce:") for k in _options(app))  # one entry, not one per dimension
        assert app.screen.entries["zoom"].key == "+"
        assert not app.screen.entries["fit"].enabled  # greyed out, not hidden
        await pilot.press("escape")
        dims = app.query_one("DimensionPanel")
        dims.cursor = dims.dims().index("projection_x_coordinate")
        dims.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert [name for name, _ in app.screen.sections] == ["Here", "Reduce", "View", "Plot"]
        assert not app.screen.entries["x:projection_x_coordinate"].enabled


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
        assert _options(app)["uncentre"] == "Undo centring" and not app.screen.entries["uncentre"].enabled
        app.screen.dismiss("centre_menu")  # longitude alone, or latitude too
        await pilot.pause()
        assert "centre" in _options(app)
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
        assert _options(app)["centre_menu"] == "Centre here…"
        app.screen.dismiss("centre_menu")
        await pilot.pause()
        assert _options(app) == {"centre": "longitude", "centre_lat": "longitude + latitude"}
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


async def test_dimension_panel_steps_wrap_round(app):
    """← on the first step goes to the last, → on the last back to the first (4 times, 6 levels)."""
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("tab")  # Variables → Dimensions, cursor on time
        assert app.viewer.pos("time") == 0
        await pilot.press("left")
        await settle(pilot, app)
        assert app.viewer.pos("time") == 3
        await pilot.press("right")
        await settle(pilot, app)
        assert app.viewer.pos("time") == 0
        await pilot.press("down", "left")  # full_levels 0 → 5
        await settle(pilot, app)
        assert app.viewer.pos("full_levels") == 5
        assert app.last_frame.request.index_dict()["full_levels"] == 5


async def test_bracket_keys_wrap_round(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("left_square_bracket")  # time 0 → 3
        await settle(pilot, app)
        assert app.viewer.pos("time") == 3
        await pilot.press("left_curly_bracket")  # full_levels 0 → 5
        await settle(pilot, app)
        assert app.viewer.pos("full_levels") == 5
        await pilot.press("right_curly_bracket")  # and back to 0
        await settle(pilot, app)
        assert app.viewer.pos("full_levels") == 0


async def test_dimension_panel_probe_wraps_round_a_plotted_dimension(app):
    """x has 7 points and the probe starts at 3: four steps right go round to 0."""
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("tab")
        dims = app.query_one("#dims")
        dims.cursor = dims.dims().index("projection_x_coordinate")
        assert app.viewer.pos("projection_x_coordinate") == 3
        await pilot.press("right", "right", "right", "right")
        await settle(pilot, app)
        assert app.viewer.pos("projection_x_coordinate") == 0


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


# --------------------------------------------------------------------------- reducing over dimensions




def _pane(app):
    return str(app.query_one("DimensionPanel").render()).splitlines()


async def test_average_over_time_from_the_dimension_menu_and_restore(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        await pilot.press("right_square_bracket")  # time 0 → 1
        await pilot.press("tab")  # Variables → Dimensions, cursor on time
        await pilot.press("enter")
        await pilot.pause()
        assert _options(app)["reduce:time"] == "Reduce time…"
        app.screen.dismiss("reduce:time")
        await pilot.pause()
        assert isinstance(app.screen, ActionMenu)  # the statistics to choose from
        assert list(_options(app)) == ["stat:mean:time", "stat:min:time", "stat:max:time", "stat:sum:time"]
        await pilot.press("enter")  # the first: mean
        await settle(pilot, app)
        assert app.viewer.reductions() == (("time", "mean"),)
        frame = app.last_frame
        assert frame.request.reduce == (("time", "mean"),)
        assert frame.header == "map y–x · time mean · full_levels 0"
        assert frame.probe_value == 1500 + 20 + 3  # the mean of time steps 0…3 at y=2, x=3
        lines = _pane(app)
        assert lines[0].split() == ["time", "mean"] and "/" not in lines[0]
        # it can't be stepped any more: ←→ on it do nothing, and [ ] step the level instead
        await pilot.press("right", "right_square_bracket")
        await settle(pilot, app)
        assert app.viewer.pos("full_levels") == 1 and app.last_frame.request.reduce == (("time", "mean"),)
        # the same menu restores it, where it was
        await pilot.press("enter")
        await pilot.pause()
        assert [name for name, _ in app.screen.sections] == ["Reduce", "Plot"]
        assert dict(app.screen.sections)["Reduce"] == [Entry("restore:time", "Restore time"),
                                                      Entry("reduce:time", "Statistic… (mean)")]
        await pilot.press("enter")
        await settle(pilot, app)
        assert app.viewer.reductions() == () and app.viewer.pos("time") == 1
        assert app.last_frame.header == "map y–x · time 2000-01-01 00:30 (T+1 h 00 min) · full_levels 1"
        assert "2/4" in _pane(app)[0]


async def test_zonal_mean_from_the_plot_menu_then_a_profile_of_it(latlon):
    app = NctuiApp(Source(latlon), "air_temperature")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.query_one(PlotView).focus()
        await pilot.press("enter")
        await pilot.pause()
        app.screen.dismiss("reduce_pick")
        await pilot.pause()
        assert _options(app)["reduce:longitude"] == "longitude"
        app.screen.dismiss("reduce:longitude")
        await pilot.pause()
        app.screen.dismiss("stat:mean:longitude")
        await settle(pilot, app)
        assert (app.viewer.kind, app.viewer.xdim, app.viewer.ydim) == ("map", "latitude", "level_height")
        assert app.last_frame.header == "section level_height–latitude · time 2000-01-01 · longitude mean"
        assert "unweighted mean" in app.last_frame.status
        await pilot.press("enter")
        await pilot.pause()
        options = _options(app)
        assert "line:longitude" not in options  # no line along a collapsed dimension
        assert options["restore:longitude"] == "Restore longitude (mean)"
        assert "reduce_pick" in options
        app.screen.dismiss("line:level_height")
        await settle(pilot, app)
        assert app.last_frame.header.startswith("profile along level_height at latitude=0 · ")
        assert app.last_frame.request.reduce == (("longitude", "mean"),)
        # the latitude menu says its mean is weighted
        app._menu_choice("reduce:latitude")
        await pilot.pause()
        assert _options(app)["stat:mean:latitude"] == "Mean (cos(lat) weights)"
        app.screen.dismiss("stat:mean:latitude")
        await settle(pilot, app)
        assert "cos(lat)-weighted mean" in app.last_frame.status


async def test_reduced_dimension_cant_go_on_an_axis_or_be_animated(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.set_reduction("time", "max")
        await settle(pilot, app)
        await pilot.press("tab")
        await pilot.press("x")  # time on the x axis: no
        await settle(pilot, app)
        assert app.viewer.xdim == "projection_x_coordinate"
        app._start_animation("time")
        assert app._anim_timer is None and "time is reduced" in app._play_error
        await pilot.press("space")  # plays the level instead
        await pilot.pause()
        assert app._anim_timer is not None and app._anim_dim == "full_levels"
        assert await wait_for(pilot, lambda: app.viewer.pos("full_levels") > 0)
        await pilot.press("space")
        await settle(pilot, app)
        assert app.last_frame.request.reduce == (("time", "max"),)


async def test_reductions_stay_when_the_variable_changes(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.set_reduction("time", "mean")
        app.set_reduction("full_levels", "max")
        await settle(pilot, app)
        app.select_var("rho")
        await settle(pilot, app)
        assert app.last_frame.request.var == "rho"
        assert app.last_frame.request.reduce == (("time", "mean"), ("half_levels", "max"))
        assert _pane(app)[2].split() == ["half_levels", "max"]


async def test_a_long_reduction_shows_progress_and_esc_cancels_it(app, monkeypatch):
    import threading

    from nctui import data

    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        before = app.last_frame.request
        monkeypatch.setattr(data, "REDUCE_BLOCK_BYTES", 1)  # a block per time step
        monkeypatch.setattr(app, "REDUCE_QUIET", 0.0)
        gate, reads = threading.Event(), []
        read = app.src._slice

        def slow(*args):
            reads.append(1)
            if len(reads) > 1:
                gate.wait(10)  # the second block hangs until the test lets go
            return read(*args)

        monkeypatch.setattr(app.src, "_slice", slow)
        app.viewer.set_pos("full_levels", 3)  # a level that hasn't been read
        app.set_reduction("time", "mean")
        assert await wait_for(pilot, lambda: "mean over time" in str(app.query_one("#topbar-left").render()))
        assert "esc cancels" in str(app.query_one("#topbar-left").render())
        assert app.last_frame.request == before  # the old plot stays up meanwhile
        await pilot.press("escape")
        gate.set()
        await settle(pilot, app)
        assert await wait_for(pilot, lambda: not app._building)
        assert app.viewer.reductions() == () and app.viewer.pos("full_levels") == 0  # back to what was drawn
        assert app.last_frame.request.reduce == ()
        assert "mean over time" not in str(app.query_one("#topbar-left").render())
        assert not app.src.reduce_cached("theta", {"full_levels": 3}, (("time", "mean"),))


async def test_restoring_while_reducing_stops_the_reduction(app, monkeypatch):
    import threading

    from nctui import data

    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        monkeypatch.setattr(data, "REDUCE_BLOCK_BYTES", 1)
        gate, reads = threading.Event(), []
        read = app.src._slice

        def slow(*args):
            reads.append(1)
            if len(reads) > 1:
                gate.wait(10)
            return read(*args)

        monkeypatch.setattr(app.src, "_slice", slow)
        app.viewer.set_pos("full_levels", 3)
        app.set_reduction("time", "mean")
        assert await wait_for(pilot, lambda: len(reads) > 1)
        app.set_reduction("time", None)
        gate.set()
        await settle(pilot, app)
        assert await wait_for(pilot, lambda: app.last_frame.request.index_dict().get("full_levels") == 3)
        assert app.last_frame.request.reduce == () and len(reads) <= 4  # the mean was dropped part way
        assert not app.src.reduce_cached("theta", {"full_levels": 3}, (("time", "mean"),))


async def test_fit_animation_range_of_a_reduced_view(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.set_reduction("time", "mean")
        app.viewer.colour.mode = "anim"
        app.view_changed()
        assert await wait_for(pilot, lambda: app.viewer.anim_ranges)
        await settle(pilot, app)
        # theta = 1000 t + 100 z + 10 y + x: the time mean over every level spans 1500 … 1500 + 546
        assert app.viewer.request().anim_range == (1500.0, 2046.0)
        assert app.last_frame.vrange == (1500.0, 2046.0)
        assert "range of all full_levels steps" in app.last_frame.stats


async def test_goto_and_settings_leave_out_a_reduced_dimension(app):
    from textual.widgets import Select

    from nctui.screens.goto import GoToScreen

    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.set_reduction("time", "mean")
        await settle(pilot, app)
        await pilot.press("g")
        await pilot.pause()
        assert isinstance(app.screen, GoToScreen)
        assert not app.screen.query("#dim-0") and app.screen.query("#dim-1")
        app.screen.query_one("#dim-1").value = "#4"
        app.screen.query_one("#dim-1").focus()
        await pilot.press("enter")
        await settle(pilot, app)
        assert app.viewer.pos("full_levels") == 4 and app.viewer.reductions() == (("time", "mean"),)
        await pilot.press("p")
        await pilot.pause()
        choices = [value for _, value in app.screen.query_one("#xdim", Select)._options]
        assert "time" not in choices and "full_levels" in choices


async def test_save_a_reduced_plot(app, tmp_path):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.set_reduction("time", "mean")
        await settle(pilot, app)
        await pilot.press("e")
        await pilot.pause()
        assert app.screen.query_one("#file").value == "./theta_timemean_full_levels0.png"
        assert "time mean" in app.screen.query_one("#title").value
        out = tmp_path / "mean.png"
        app.screen.query_one("#file").value = str(out)
        await pilot.press("enter")
        assert await wait_for(pilot, out.exists)


async def test_every_dimension_reduced_shows_one_number_and_can_be_restored(app):
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        for d in app.viewer.v.dims:
            app.set_reduction(d, "min")
        await settle(pilot, app)
        assert app.last_frame.error is None and app.last_frame.probe_value == 0.0
        assert app.last_frame.header == "value · time min · full_levels min · y min · x min"
        app.query_one(PlotView).focus()
        await pilot.press("enter")
        await pilot.pause()
        options = _options(app)
        assert options["restore:time"] == "Restore time (min)" and not any(k.startswith("line:") for k in options)
        app.screen.dismiss("restore:time")
        await settle(pilot, app)
        assert app.last_frame.header == "time series along time · full_levels min · y min · x min"
        await pilot.press("g", "escape", "p", "escape")  # the popups cope with one dimension left
        await settle(pilot, app)
        assert app.last_frame.error is None


async def test_cancelling_an_animation_save_stops_its_colour_range_scan(app, tmp_path, monkeypatch):
    """The fit-animation range of a reduced view is a long read: cancel must reach it."""
    import threading

    from nctui import data

    class Dialog:
        said: list = []

        def progress(self, text):
            self.said.append(text)

        def finished(self, ok, text):
            self.said.append((ok, text))

    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, app)
        app.set_reduction("time", "mean")
        await settle(pilot, app)
        monkeypatch.setattr(data, "REDUCE_BLOCK_BYTES", 8 * 5 * 7)  # one time step per block
        gate, reads = threading.Event(), []
        read = app.src._slice

        def slow(*args):
            reads.append(1)
            gate.wait(10)
            return read(*args)

        monkeypatch.setattr(app.src, "_slice", slow)
        app.viewer.colour.mode = "anim"  # not scanned yet: the save has to scan
        out = tmp_path / "a.gif"
        dialog = Dialog()
        app.save_animation(str(out), "", (3, 2), 40, 4, [0, 1, 2], dialog)
        assert await wait_for(pilot, lambda: reads)
        app.cancel_save()
        gate.set()
        assert await wait_for(pilot, lambda: not app._save_worker.is_running)
        assert len(reads) <= 2  # not the 20-odd blocks of every level
        assert not out.exists() and dialog.said == ["scanning all full_levels steps…"]
