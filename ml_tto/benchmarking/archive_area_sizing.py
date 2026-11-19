"""Estimate per-second data volume of the archiver for each beamline area.

For each area, does ONE ``caget_many`` to inspect sample dtypes/shapes, then
reports:

- resolved PV count (accessor set)
- bytes per sample (sum of each PV's first-read array nbytes)
- estimated achieved rate from the 0.15 ms-per-PV scaling law
- estimated MB/s at that rate

The scaling law was calibrated on the scalar-only GUN accessor set, so
areas carrying large image/waveform PVs will tend to be slower per PV in
practice -- treat the Hz estimate as an upper bound and the MB/s as a
lower bound of the real data pressure.

Usage on prod::

    python archive_area_sizing.py --areas GUN HTR DIAG0

Expects ``slac_db``, ``pyepics``, and ``ml_tto`` to be importable.
"""

from __future__ import annotations

import argparse

import numpy as np

from ml_tto.archiving import build_pv_list


MS_PER_PV = 0.15


def _bytes_for_value(v) -> int:
    if v is None:
        return 0
    if isinstance(v, (bytes, str)):
        return len(v) if isinstance(v, bytes) else len(v.encode("utf-8"))
    return int(np.asarray(v).nbytes)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} TB"


def _measure_area(area: str) -> dict:
    from epics import caget_many

    pv_loc = build_pv_list(area)
    pvs = [p for p, _ in pv_loc]
    if not pvs:
        return {"area": area, "n_pvs": 0, "missing": 0,
                "bytes_per_sample": 0, "est_rate_hz": float("nan"), "est_MBps": 0.0}
    first = caget_many(pvs, timeout=10.0)
    bytes_per_sample = sum(_bytes_for_value(v) for v in first if v is not None)
    missing = sum(1 for v in first if v is None)
    est_cycle_s = MS_PER_PV * 1e-3 * len(pvs)
    est_rate_hz = 1.0 / est_cycle_s if est_cycle_s > 0 else float("nan")
    return {
        "area": area,
        "n_pvs": len(pvs),
        "missing": missing,
        "bytes_per_sample": bytes_per_sample,
        "est_rate_hz": est_rate_hz,
        "est_MBps": est_rate_hz * bytes_per_sample / 1e6,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", nargs="+", required=True)
    args = ap.parse_args()

    header = f"{'area':<10} {'PVs':>6} {'miss':>5} {'sample':>12} {'est Hz':>8} {'est MB/s':>10}"
    print(header)
    print("-" * len(header))
    for area in args.areas:
        try:
            r = _measure_area(area)
        except Exception as e:
            print(f"{area:<10}  ERROR: {e}")
            continue
        print(
            f"{r['area']:<10} {r['n_pvs']:>6} {r['missing']:>5}"
            f" {_fmt_bytes(r['bytes_per_sample']):>12}"
            f" {r['est_rate_hz']:>8.1f}"
            f" {r['est_MBps']:>10.3f}"
        )


if __name__ == "__main__":
    main()
