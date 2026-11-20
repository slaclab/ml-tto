# ml_tto.archiving

Area-based EPICS PV archiver. Given one or more beamline areas defined in
[`slac-db`](https://github.com/slaclab/slac-db) (plus any extra PVs you
pass explicitly), reads the PV values with `epics.caget_many` and writes
them to a single HDF5 file following
[Eric Cropp's data-sharing SPEC v0.1.2](https://github.com/ericcropp/Data_Sharing).

## Quick start

```python
from ml_tto.archiving import snapshot, continuous

# one-shot read
snapshot(
    areas="GUN",
    extra_pvs=["OTRS:IN20:571:XRMS"],
    path="/data/gb/gun_snap.h5",
    metadata={"operator": "gb", "purpose": "baseline"},
)

# fixed-rate stream
continuous(
    areas=["GUN", "HTR"],
    path="/data/gb/stream.h5",
    rate_hz=2.0,
    n_samples=600,                   # or duration_s=300
)
```

Exactly one of `n_samples` / `duration_s` must be set for `continuous`.
Both entry points return the output `Path`.

### Appending snapshots to an existing file

Pass an open, writable `h5py.File` (or `h5py.Group`) as `group=` instead
of `path=`. Each call adds a new batch and appends its ID to the group's
`@IDs`; the new batch group is returned.

```python
import h5py
from ml_tto.archiving import snapshot

with h5py.File("/data/gb/gun_snaps.h5", "a") as f:
    snapshot("GUN", group=f, notes="before quad change")
    # ... change something ...
    snapshot("GUN", group=f, notes="after quad change")
```

The group is treated as the spec root, so pass the file itself to get a
spec-conformant file. An existing `@Data_Standard_Version` that differs
from `0.1.2` is rejected before any PVs are read. If a snapshot fails,
the file is left as it was (no partial batch, `@IDs` unchanged).

## What gets archived

For each area, PVs come from `slac_db.device.get_all_accessors(device)` —
the curated accessor→PV mapping, not the full address book. Extras are
appended verbatim with `location=""`. Dedup is applied across overlapping
areas; first area to introduce a PV wins the `@location` tag.

## File layout (Cropp SPEC v0.1.2)

```
<file>.h5
├── @Data_Standard_Version = "0.1.2"
├── @IDs = [<batch_id>]
└── <batch_id>/
    ├── @batch_dims = (N,)
    ├── @run_information_source
    ├── @run_information_date
    ├── @run_information_notes             # JSON: metadata, missing_pvs,
    │                                      # requested_rate_hz, achieved_rate_hz
    └── observables/
        ├── Timestamp                      # (N,) ISO-8601 UTF-8
        └── <PV>                           # (N, *feat) float64 or vlen UTF-8
```

String-valued PVs (status/enum PVs like `*:ACCESS`, `*:ABORT`) are stored
as vlen UTF-8; numeric scalars and waveforms as float64 with NaN padding
for missing reads. The dataset name is the full PV; `@location` carries
the originating area.

## Rate limits

`rate_hz` is a **target**, not a guarantee. The loop is bounded by
`caget_many` round-trip.

Measured on LCLS (GUN accessor set) at `rate_hz=1000` (no pacing):

| config              |  PVs  | achieved rate | cycle time |
|---------------------|-------|---------------|------------|
| 25% subset          |   19  | 339 Hz        | 3.0 ms     |
| 50% subset          |   38  | 177 Hz        | 5.6 ms     |
| 100% accessor set   |   76  | 91 Hz         | 11 ms      |
| full address book   | 10012 | 0.53 Hz       | 1900 ms    |

**Scaling law:** cycle time ≈ **0.15 ms per PV**, linear, near-zero fixed
overhead. Safe sustained rate (half the ceiling, for headroom) is:

> `safe_Hz ≈ 1 / (0.0003 × N_PVs)`

Beam rate (120 Hz) is **unreachable** from Python / `caget_many`. For
beam-synchronous archiving use EPICS monitors or BSA.

Each output file carries `achieved_rate_hz` and `requested_rate_hz` in
`run_information_notes` so callers can see what they actually got.

## Measured per-area data volume (LCLS, accessor sets)

Produced with `examples/archive_area_sizing.py` — one `caget_many` per
area, Hz estimated from the 0.15 ms/PV law, MB/s = est_Hz × bytes/sample:

| area  | PVs  | miss | sample size | est Hz | est MB/s |
|-------|------|------|-------------|--------|----------|
| GUN   |   76 |   0  |    600 B    |  87.7  |   0.053  |
| HTR   |  320 |   6  |  427.1 KB   |  20.8  |   9.111  |
| DIAG0 |  274 |   0  |    5.5 MB   |  24.3  | 139.302  |

**Caveats on these numbers**

- The Hz estimate is calibrated on scalar-only PVs (GUN). Areas with
  cameras (HTR, DIAG0) have large waveforms where per-PV time is likely
  higher than 0.15 ms, so **est Hz is an upper bound** and **est MB/s is
  a lower bound** on real-time data pressure.
- `miss` counts PVs whose first read returned None (not connected / stale
  slac_db entry). Non-zero `miss` is a prompt to eyeball the PV list.
- `sample size` is total bytes per sample across every resolved PV.

## Examples

Three ready-to-run scripts live in `examples/`:

| script                        | purpose                                              |
|-------------------------------|------------------------------------------------------|
| `archive_gun_smoke.py`        | end-to-end smoke test: resolve, snapshot, short stream |
| `archive_rate_benchmark.py`   | unpaced sweep; measures the Hz ceiling for an area    |
| `archive_area_sizing.py`      | per-area table of PV count, sample size, est MB/s     |

All three take `--area`/`--areas` and `--outdir`. See each script's
docstring for details.

## Durability

Datasets are preallocated and written per-sample (`d[i] = value`), but
not flushed per sample — HDF5 buffers go to disk when the file closes
(`finally: f.close()`). A clean exit (success or Python exception) is
safe; a `kill -9` or power loss can drop the tail of the samples. If you
need per-sample durability, open an issue and we'll add a `flush_every`
kwarg.

## Caveats

- `@units = ""` on every dataset. `slac_db` accessor rows don't expose
  units uniformly; post-process the file if you need them.
- `@bin_size = 1.0`, `@offset = 0.0` for waveform/image PVs — placeholders
  required by the Cropp spec when `num_feature_dims > 0`. Not physical
  calibration.
- PVs that return None on the first read are dropped (not written) and
  listed under `run_information_notes.missing_pvs`. Later shape mismatches
  get NaN-filled with a warning; the loop keeps going.
