"""Measure the practical rate ceiling of ml_tto.archiving on a prod host.

Approach: request a very high rate so the loop never sleeps, run N samples,
read back ``achieved_rate_hz`` and ``requested_rate_hz`` from the file's
``run_information_notes``. Repeat for a few configurations so we can show
how the ceiling moves with PV count and waveform payload.

Usage on prod::

    python archive_rate_benchmark.py --outdir /tmp/gb_archive_bench

What it reports, per configuration:

- resolved PV count
- achieved rate (Hz) and average cycle time (s)
- fraction of cycles that slipped

Three configurations by default:

1. Accessor-only set for GUN (what the real archiver will use).
2. Same set, but with any PVs whose first-read value has a feature
   dimension > 0 (i.e. waveforms/images) dropped -- isolates the cost
   of the big array PVs.
3. Scalar-only sweep at 25 %, 50 %, 100 % of the scalar set, to show
   how cycle time scales with PV count.

No extra pacing -- we set rate_hz to 1000 so time.sleep() never fires.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from ml_tto.archiving import build_pv_list, continuous


UNPACED_HZ = 1000.0


def _read_achieved(path: Path) -> tuple[float, int]:
    with h5py.File(path, "r") as f:
        (bid,) = [s.decode() if isinstance(s, bytes) else s for s in f.attrs["IDs"]]
        batch = f[bid]
        notes = json.loads(batch.attrs["run_information_notes"])
        n_pvs = len([k for k in batch["observables"].keys() if k != "Timestamp"])
    return notes.get("achieved_rate_hz", float("nan")), n_pvs


def _run_and_report(
    label: str, extra_pvs: list[str], outdir: Path, n_samples: int, area: str | None
) -> None:
    out = outdir / f"bench_{label}.h5"
    print(f"\n=== {label} ===")
    if area is None and not extra_pvs:
        print("  (empty set, skipping)")
        return
    continuous(
        areas=area if area is not None else [],
        extra_pvs=extra_pvs,
        path=out,
        rate_hz=UNPACED_HZ,
        n_samples=n_samples,
    )
    hz, n_pvs = _read_achieved(out)
    cycle_s = 1.0 / hz if hz > 0 else float("inf")
    print(f"  PVs tracked        : {n_pvs}")
    print(f"  achieved rate (Hz) : {hz:.3f}")
    print(f"  avg cycle time (s) : {cycle_s:.3f}")


def _classify_pvs(area: str) -> tuple[list[str], list[str]]:
    """Return (scalar_pvs, waveform_pvs) based on a single caget read."""
    from epics import caget_many

    pv_loc = build_pv_list(area)
    pvs = [p for p, _ in pv_loc]
    first = caget_many(pvs, timeout=10.0)
    scalars, waveforms = [], []
    for pv, v in zip(pvs, first):
        if v is None:
            continue
        arr = np.asarray(v)
        if arr.ndim == 0 or (arr.ndim == 1 and arr.size <= 1):
            scalars.append(pv)
        else:
            waveforms.append(pv)
    return scalars, waveforms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--area", default="GUN")
    ap.add_argument("--n-samples", type=int, default=10)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Classifying PVs for area {args.area!r} (one caget_many to inspect shapes) ...")
    scalars, waveforms = _classify_pvs(args.area)
    print(f"  scalars  : {len(scalars)}")
    print(f"  waveforms: {len(waveforms)}")

    _run_and_report(
        "full_accessor_set",
        extra_pvs=scalars + waveforms,
        outdir=args.outdir,
        n_samples=args.n_samples,
        area=None,
    )
    _run_and_report(
        "scalars_only",
        extra_pvs=scalars,
        outdir=args.outdir,
        n_samples=args.n_samples,
        area=None,
    )
    for frac in (0.25, 0.5, 1.0):
        k = max(1, int(len(scalars) * frac))
        _run_and_report(
            f"scalars_{int(frac*100):03d}pct_{k}",
            extra_pvs=scalars[:k],
            outdir=args.outdir,
            n_samples=args.n_samples,
            area=None,
        )


if __name__ == "__main__":
    main()
