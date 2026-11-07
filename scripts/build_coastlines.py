"""Convert Natural Earth's 110 m coastline shapefile into the array file nctui ships.

Natural Earth data is in the public domain (https://www.naturalearthdata.com/).
Download ne_110m_coastline.zip (for example from
https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip), then:

    pixi run python scripts/build_coastlines.py path/to/ne_110m_coastline.zip

The result, src/nctui/data/ne_110m_coastline.npz, holds two float32 arrays,
``lon`` and ``lat``, with NaN separating the lines.
"""

from __future__ import annotations

import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "src" / "nctui" / "data" / "ne_110m_coastline.npz"


def read_polylines(shp: bytes) -> list[np.ndarray]:
    code, = struct.unpack(">i", shp[0:4])
    if code != 9994:
        raise ValueError("not a shapefile")
    shape_type, = struct.unpack("<i", shp[32:36])
    if shape_type != 3:
        raise ValueError(f"expected polylines (type 3), got type {shape_type}")
    lines = []
    pos = 100
    while pos + 8 <= len(shp):
        _, length_words = struct.unpack(">ii", shp[pos:pos + 8])
        content = shp[pos + 8:pos + 8 + 2 * length_words]
        pos += 8 + 2 * length_words
        rec_type, = struct.unpack("<i", content[0:4])
        if rec_type == 0:
            continue
        nparts, npoints = struct.unpack("<ii", content[36:44])
        parts = list(struct.unpack(f"<{nparts}i", content[44:44 + 4 * nparts]))
        start = 44 + 4 * nparts
        pts = np.frombuffer(content[start:start + 16 * npoints], dtype="<f8").reshape(npoints, 2)
        for a, b in zip(parts, parts[1:] + [npoints], strict=True):
            lines.append(pts[a:b])
    return lines


def main(zip_path: str) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".shp"))
        lines = read_polylines(zf.read(name))
    lon, lat = [], []
    for line in lines:
        lon.extend(line[:, 0].tolist() + [np.nan])
        lat.extend(line[:, 1].tolist() + [np.nan])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, lon=np.asarray(lon, np.float32), lat=np.asarray(lat, np.float32))
    print(f"wrote {OUT}: {len(lines)} lines, {len(lon) - len(lines)} points")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
