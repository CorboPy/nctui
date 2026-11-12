import pytest

from nctui.cli import main


def test_plot_prints_a_figure(cartesian, capsys):
    code = main(["plot", str(cartesian), "-v", "theta", "--at", "time=#1", "--width", "80",
                 "--height", "20", "--no-color"])
    out = capsys.readouterr().out
    assert code == 0
    assert "theta · air_potential_temperature · K" in out
    assert "time 2000-01-01 00:30" in out
    assert "▀" in out


def test_plot_line(cartesian, capsys):
    code = main(["plot", str(cartesian), "-v", "theta", "--x", "full_levels", "--width", "60",
                 "--height", "15", "--no-color"])
    assert code == 0
    assert "profile along full_levels" in capsys.readouterr().out


def test_unknown_variable_suggests(cartesian, capsys):
    with pytest.raises(SystemExit):
        main(["plot", str(cartesian), "-v", "thetaa"])
    assert "Did you mean: theta" in capsys.readouterr().err


def test_missing_file(capsys):
    with pytest.raises(SystemExit):
        main(["/no/such/file.nc"])
    assert "no such file" in capsys.readouterr().err


def test_bad_at(cartesian, capsys):
    assert main(["plot", str(cartesian), "--at", "nonsense"]) == 1


def _plot(capsys, *args):
    code = main(["plot", *map(str, args), "--width", "80", "--height", "20", "--no-color"])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_plot_mean_over_a_dimension(cartesian, capsys):
    code, out, _ = _plot(capsys, cartesian, "-v", "theta", "--mean", "time", "--at", "full_levels=#1")
    assert code == 0
    assert "map y–x · time mean · full_levels 1" in out
    assert "min 1600  max 1646" in out  # 1000 × the mean time index (1.5) + 100 × level 1
    assert "unweighted mean" in out


def test_plot_other_statistics_and_a_series(cartesian, capsys):
    code, out, _ = _plot(capsys, cartesian, "-v", "theta", "--max", "projection_y_coordinate",
                         "--max", "projection_x_coordinate", "--min", "full_levels")
    assert code == 0
    assert "time series along time · full_levels min · y max · x max" in out
    assert "min 46  max 3046" in out
    code, out, _ = _plot(capsys, cartesian, "-v", "theta", "--sum", "time", "--x", "full_levels")
    assert code == 0 and "profile along full_levels" in out and "time sum" in out


def test_plot_weighted_domain_mean(latlon, capsys):
    code, out, _ = _plot(capsys, latlon, "-v", "toa_flux", "--mean", "latitude", "--mean", "longitude")
    assert code == 0
    assert "time series along time · latitude mean · longitude mean" in out
    assert "min 1  max 1  mean 1" in out
    assert "→ 1 W m-2 · cos(lat)-weighted mean" in " ".join(out.split())  # the status line wraps at 80


def test_plot_rejects_bad_reductions(cartesian, capsys):
    code, _, err = _plot(capsys, cartesian, "-v", "theta", "--mean", "half_levels")
    assert code == 1 and "theta has no dimension half_levels" in err
    code, _, err = _plot(capsys, cartesian, "-v", "theta", "--mean", "time", "--max", "time")
    assert code == 1 and "time" in err and "one statistic" in err
    code, _, err = _plot(capsys, cartesian, "-v", "theta", "--mean", "time", "--at", "time=#1")
    assert code == 1 and "time is reduced" in err
    code, _, err = _plot(capsys, cartesian, "-v", "theta", "--mean", "time", "--x", "time")
    assert code == 1 and "restore it first" in err


def test_plot_mean_over_joined_files(split_files, capsys):
    code, out, _ = _plot(capsys, *split_files, "-v", "theta", "--mean", "time")
    assert code == 0
    assert "map y–x · time mean · full_levels 0" in out
    assert "min 2500  max 2546" in out  # 1000 × the mean of time steps 0…5, across the three files
