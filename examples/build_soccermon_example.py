"""Build the example dataset: mean-maximal speed curves of one SoccerMon player.

SoccerMon (Midoglu et al. 2024, Scientific Data 11:553; Zenodo 10033832,
CC-BY 4.0) holds 10 Hz GNSS recordings of two elite women's football teams.
For one pseudonymised player, every recording is read by HTTP range requests
from the archives on Zenodo, reduced to its mean-maximal speed curve, and
dropped. Output: src/critpower/data/soccermon_mms.csv (date, duration, speed).

    python examples/build_soccermon_example.py
"""

from __future__ import annotations

import csv
import io
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from critpower.field import mean_maximal  # noqa: E402

PLAYER = "TeamA-f20565cc-df22-46a7-aa97-af8ed00601d7"
ARCHIVES = ["objective-TeamA-2020.zip", "objective-TeamA-2021.zip"]
URL = "https://zenodo.org/records/10033832/files/{}?download=1"
DURATIONS = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200]
OUT = Path(__file__).resolve().parents[1] / "src" / "critpower" / "data" / "soccermon_mms.csv"


def _get(url, start, end):
    for k in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(f"cannot read {url} {start}-{end}")


def members(url):
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        size = int(r.headers["Content-Length"])
    tail = _get(url, size - 65536 - 22, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        _, cd_size, cd_off = struct.unpack("<QQQ", tail[j + 32:j + 56])
    cd, p, out = _get(url, cd_off, cd_off + cd_size - 1), 0, []
    while cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode()
        extra, q = cd[p + 46 + nlen:p + 46 + nlen + xlen], 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, r = [], q + 4
                for flag in (usize, csize, off):
                    if flag == 0xFFFFFFFF:
                        vals.append(struct.unpack("<Q", extra[r:r + 8])[0]); r += 8
                    else:
                        vals.append(flag)
                usize, csize, off = vals
            q += 4 + hlen
        out.append({"name": name, "method": method, "csize": csize, "offset": off})
        p += 46 + nlen + xlen + clen
    return out


def curve(url, m):
    head = _get(url, m["offset"], m["offset"] + 29)
    nlen, xlen = struct.unpack("<HH", head[26:30])
    start = m["offset"] + 30 + nlen + xlen
    raw = _get(url, start, start + m["csize"] - 1)
    raw = zlib.decompress(raw, -15) if m["method"] == 8 else raw
    df = pq.read_table(io.BytesIO(raw), columns=["time", "speed"]).to_pandas().drop_duplicates("time")
    import pandas as pd
    t = pd.to_timedelta(df["time"].astype(str)).dt.total_seconds().to_numpy()
    v = np.clip(df["speed"].to_numpy(dtype=float), 0, 10.0)  # GNSS spikes above 10 m/s are not running
    return m["name"].split("/")[-1][:10], mean_maximal(t, v, DURATIONS, dt=0.1)


def main():
    rows = []
    for arc in ARCHIVES:
        url = URL.format(arc)
        ms = [m for m in members(url) if PLAYER in m["name"] and m["name"].endswith(".parquet")]
        names = [m["name"] for m in ms]
        ms = [m for m in ms if names.count(m["name"]) == 1]  # duplicate names are ambiguous: skipped
        print(arc, len(ms), "recordings", flush=True)
        with ThreadPoolExecutor(16) as ex:
            for fut in as_completed([ex.submit(curve, url, m) for m in ms]):
                day, c = fut.result()
                rows += [(day, d, round(v, 4)) for d, v in c.items()]
    rows.sort()
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "duration_s", "speed_ms"])
        w.writerows(rows)
    print(OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
