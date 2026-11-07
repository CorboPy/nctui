"""Small synthetic netCDF files for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

EXAMPLE_DATA = Path(__file__).resolve().parent.parent / "example_data"


def make_cartesian(path: Path, nt=4, nz=6, ny=5, nx=7, t0=0.0, calendar="standard") -> Path:
    """A 4-D t/z/y/x file shaped like the LFRic examples, plus a half-level variable."""
    t = t0 + np.arange(nt) * 1800.0
    z = np.arange(nz, dtype=np.float32)
    zh = np.arange(nz - 1, dtype=np.float32) + 0.5
    y = np.arange(ny) * 100.0
    x = np.arange(nx) * 100.0
    tt, zz, yy, xx = np.meshgrid(np.arange(nt), np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    # value encodes the (global) time index, level, y and x: 1000*t + 100*z + 10*y + x
    theta = (1000 * (tt + t0 / 1800.0) + 100 * zz + 10 * yy + xx).astype(np.float32)
    rho = theta[:, :-1] * 2
    incr = (xx - nx / 2).astype(np.float32)  # straddles zero
    ds = xr.Dataset(
        {
            "theta": (("time", "full_levels", "projection_y_coordinate", "projection_x_coordinate"), theta,
                      {"long_name": "air_potential_temperature", "units": "K"}),
            "rho": (("time", "half_levels", "projection_y_coordinate", "projection_x_coordinate"), rho,
                    {"long_name": "air_density", "units": "kg m-3"}),
            "dtheta": (("time", "full_levels", "projection_y_coordinate", "projection_x_coordinate"), incr,
                       {"units": "K"}),
            "time_bnds": (("time", "bnds"), np.stack([t, t + 1800], axis=1)),
            "forecast_period": (("time",), t - t0 + 1800, {"units": "seconds"}),
        },
        coords={
            "time": ("time", t, {"axis": "T", "units": "seconds since 2000-01-01 00:00:00",
                                 "calendar": calendar, "standard_name": "time", "bounds": "time_bnds"}),
            "full_levels": ("full_levels", z),
            "half_levels": ("half_levels", zh),
            "projection_y_coordinate": ("projection_y_coordinate", y, {"units": "m"}),
            "projection_x_coordinate": ("projection_x_coordinate", x, {"units": "m"}),
        },
    )
    ds.to_netcdf(path)
    return path


def make_latlon(path: Path, nt=3, nz=4, nlat=9, nlon=12) -> Path:
    t = np.arange(nt) * 10.0
    lev = np.arange(nz) * 1000.0
    lat = np.linspace(-80, 80, nlat).astype(np.float32)
    lon = (np.arange(nlon) * 360.0 / nlon + 15).astype(np.float32)
    data = np.random.default_rng(0).normal(280, 5, (nt, nz, nlat, nlon)).astype(np.float32)
    data[0, 0, 0, 0] = np.nan
    flux = np.ones((nt, nlat, nlon), dtype=np.float32)
    ds = xr.Dataset(
        {
            "air_temperature": (("time", "level_height", "latitude", "longitude"), data, {"units": "K"}),
            "toa_flux": (("time", "latitude", "longitude"), flux, {"units": "W m-2"}),
        },
        coords={
            "time": ("time", t, {"axis": "T", "units": "days since 2000-01-01 00:00:00", "calendar": "360_day"}),
            "level_height": ("level_height", lev, {"axis": "Z", "units": "m", "positive": "up"}),
            "latitude": ("latitude", lat, {"axis": "Y", "units": "degrees_north"}),
            "longitude": ("longitude", lon, {"axis": "X", "units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path)
    return path


def make_pressure(path: Path) -> Path:
    p = np.array([1000.0, 850.0, 500.0, 250.0])
    lat = np.linspace(-60, 60, 5)
    data = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    ds = xr.Dataset(
        {"ta": (("plev", "lat"), data, {"units": "K"})},
        coords={"plev": ("plev", p, {"units": "hPa"}), "lat": ("lat", lat, {"units": "degrees_north"})},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture
def cartesian(tmp_path) -> Path:
    return make_cartesian(tmp_path / "cart.nc")


@pytest.fixture
def latlon(tmp_path) -> Path:
    return make_latlon(tmp_path / "latlon.nc")


@pytest.fixture
def split_files(tmp_path) -> list[Path]:
    """Three files from one run, given out of order."""
    a = make_cartesian(tmp_path / "c.nc", nt=2, t0=2 * 2 * 1800.0)
    b = make_cartesian(tmp_path / "a.nc", nt=2, t0=0.0)
    c = make_cartesian(tmp_path / "b.nc", nt=2, t0=2 * 1800.0)
    return [a, b, c]
