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
