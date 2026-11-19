"""Smoke-test for ml_tto.archiving against the GUN area on prod.

Run on a control-room host with EPICS CA access and slac_db installed:

    python examples/archive_gun_smoke.py --outdir /tmp

What it does:

1. ``build_pv_list("GUN")`` -- prints the resolved PV count.
2. ``snapshot("GUN", ...)`` -- one read, writes gun_snapshot.h5, prints a
   short structure summary.
3. A short ``continuous("GUN", rate_hz=1.0, n_samples=10)`` -- writes
   gun_stream.h5 and prints requested vs achieved rate so we can see
   what the caget round-trip actually supports for the GUN PV list.

Nothing here is destructive -- only caget + local file writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from ml_tto.archiving import build_pv_list, continuous, snapshot


def _summarize(path: Path) -> None:
    with h5py.File(path, "r") as f:
        ids = [s.decode() if isinstance(s, bytes) else s for s in f.attrs["IDs"]]
        batch = f[ids[0]]
        obs = batch["observables"]
        pvs = [k for k in obs.keys() if k != "Timestamp"]
        notes = json.loads(batch.attrs["run_information_notes"])
        print(f"  file           : {path}")
        print(f"  batch_dims     : {tuple(batch.attrs['batch_dims'])}")
        print(f"  pv datasets    : {len(pvs)}")
        if pvs:
            shapes = {obs[pv].shape for pv in pvs}
            print(f"  sample shapes  : {sorted(map(str, shapes))}")
        print(f"  missing on 1st read: {len(notes.get('missing_pvs', []))}")
        if "requested_rate_hz" in notes:
            print(f"  requested_rate_hz: {notes['requested_rate_hz']}")
        if "achieved_rate_hz" in notes:
            print(f"  achieved_rate_hz : {notes['achieved_rate_hz']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--area", default="GUN")
    ap.add_argument("--rate-hz", type=float, default=1.0)
    ap.add_argument("--n-samples", type=int, default=10)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Resolving PVs for area {args.area!r} via slac_db ...")
    pv_loc = build_pv_list(args.area)
    print(f"  resolved {len(pv_loc)} PVs")
    for pv, loc in pv_loc[:5]:
        print(f"    {loc or '<extra>':<8}  {pv}")
    if len(pv_loc) > 5:
        print(f"    ... (+{len(pv_loc) - 5} more)")

    print(f"\n[2/3] Snapshot -> {args.outdir / 'gun_snapshot.h5'}")
    snap_path = snapshot(
        args.area,
        path=args.outdir / "gun_snapshot.h5",
        run_source="LCLS",
        notes="gun-area archiver smoke-test",
        metadata={"test": "smoke", "area": args.area},
    )
    _summarize(snap_path)

    print(
        f"\n[3/3] Continuous rate_hz={args.rate_hz}, n_samples={args.n_samples} "
        f"-> {args.outdir / 'gun_stream.h5'}"
    )
    stream_path = continuous(
        args.area,
        path=args.outdir / "gun_stream.h5",
        rate_hz=args.rate_hz,
        n_samples=args.n_samples,
        run_source="LCLS",
        notes="gun-area continuous smoke-test",
        metadata={"test": "smoke", "area": args.area},
    )
    _summarize(stream_path)


if __name__ == "__main__":
    main()
