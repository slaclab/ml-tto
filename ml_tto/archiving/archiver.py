"""Area-based PV archiver.

Resolves one or more beamline areas (via :mod:`slac_db`) to a flat PV list,
optionally appends user-supplied extra PVs, then reads the PVs with
``epics.caget_many`` either once (:func:`snapshot`) or at a fixed cadence
(:func:`continuous`). Results land in a single HDF5 file following the
Cropp data-sharing standard v0.1.2
(https://github.com/ericcropp/Data_Sharing).

Rate limits (measured on LCLS GUN accessor set): cycle time is bounded
by ``caget_many`` at roughly **0.15 ms per PV**, giving a safe sustained
rate of ``1 / (0.0003 * N_PVs)`` Hz. Beam rate 120 Hz is unreachable --
use EPICS monitors for that. Achieved rate is written to
``run_information_notes.achieved_rate_hz`` in every output file.

Units note: ``@units`` is written as ``""`` because slac_db accessor rows
do not expose units uniformly. Post-process the file if units are needed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Iterable, Union

import h5py
import numpy as np

_DATA_STANDARD_VERSION = "0.1.2"

logger = logging.getLogger(__name__)


def _isotime() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .astimezone()
        .isoformat(timespec="microseconds")
    )


def _as_area_list(areas: Union[str, Iterable[str]]) -> list[str]:
    if isinstance(areas, str):
        return [areas]
    return list(areas)


def _resolve_pvs(
    areas: Union[str, Iterable[str]],
    extra_pvs: Iterable[str],
    include_inactive: bool,
) -> list[tuple[str, str]]:
    """Return a dedup list of ``(pv, location)`` pairs.

    First occurrence wins (preserves the area that introduced the PV).
    Extras are appended last with ``location=""``.
    """
    import slac_db.device as _sdb

    seen: dict[str, str] = {}
    for area in _as_area_list(areas):
        for device in _sdb.get_devices(area=area, include_inactive=include_inactive):
            for pv in _sdb.get_all_accessors(device).values():
                if pv not in seen:
                    seen[pv] = area
    for pv in extra_pvs:
        if pv not in seen:
            seen[pv] = ""
    return list(seen.items())


def build_pv_list(
    areas: Union[str, Iterable[str]],
    extra_pvs: Iterable[str] = (),
    include_inactive: bool = False,
) -> list[tuple[str, str]]:
    """Public wrapper around :func:`_resolve_pvs`."""
    return _resolve_pvs(areas, extra_pvs, include_inactive)


def _value_shape(v) -> tuple[int, ...]:
    if v is None:
        return ()
    arr = np.asarray(v)
    if arr.ndim == 0:
        return ()
    return arr.shape


def _nan_block(shape: tuple[int, ...]):
    if shape == ():
        return np.nan
    return np.full(shape, np.nan, dtype=np.float64)


def _is_string_value(v) -> bool:
    if isinstance(v, (bytes, str)):
        return True
    arr = np.asarray(v)
    return arr.dtype.kind in ("U", "S", "O")


def _coerce_string(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, str):
        return v
    return str(v)


def _missing_fill(d: h5py.Dataset):
    """Default value to write when a PV read is missing or malformed."""
    if h5py.check_string_dtype(d.dtype) is not None:
        return ""
    return _nan_block(d.shape[1:])


def _new_batch_id(source: str) -> str:
    return hashlib.md5(f"{source}|{_isotime()}".encode()).hexdigest()


def _open_file_and_batch(
    path: Path,
    batch_id: str,
    n_samples: int,
    source: str,
    notes: str,
    metadata: dict | None,
    missing_pvs: list[str],
    requested_rate_hz: float | None,
    duration_s: float | None = None,
) -> tuple[h5py.File, h5py.Group]:
    f = h5py.File(path, "w")
    f.attrs["Data_Standard_Version"] = _DATA_STANDARD_VERSION
    f.attrs["IDs"] = np.array([batch_id], dtype=h5py.string_dtype("utf-8"))

    batch = f.create_group(batch_id)
    batch.attrs["Data_Standard_Version"] = _DATA_STANDARD_VERSION
    batch.attrs["ID"] = batch_id
    batch.attrs["batch_dims"] = np.array([n_samples], dtype=np.int64)
    batch.attrs["run_information_source"] = source
    batch.attrs["run_information_date"] = datetime.date.today().isoformat()

    note_payload = {"notes": notes}
    if metadata:
        note_payload["metadata"] = metadata
    if missing_pvs:
        note_payload["missing_pvs"] = missing_pvs
    if requested_rate_hz is not None:
        note_payload["requested_rate_hz"] = requested_rate_hz
    if duration_s is not None:
        note_payload["duration_s_requested"] = duration_s
    batch.attrs["run_information_notes"] = json.dumps(note_payload)
    return f, batch


def _create_pv_dataset(
    obs: h5py.Group,
    pv: str,
    location: str,
    n_samples: int,
    first_value,
) -> h5py.Dataset:
    if _is_string_value(first_value):
        sample_shape: tuple[int, ...] = ()
        dtype = h5py.string_dtype("utf-8")
    else:
        sample_shape = _value_shape(first_value)
        dtype = np.float64
    dset = obs.create_dataset(
        pv,
        shape=(n_samples, *sample_shape),
        maxshape=(None, *sample_shape),
        dtype=dtype,
    )
    dset.attrs["location"] = location
    dset.attrs["read_only"] = True
    dset.attrs["num_feature_dims"] = len(sample_shape)
    dset.attrs["units"] = ""
    if sample_shape:
        dset.attrs["bin_size"] = 1.0
        dset.attrs["offset"] = 0.0
    return dset


def _create_timestamp_dataset(obs: h5py.Group, n_samples: int) -> h5py.Dataset:
    dset = obs.create_dataset(
        "Timestamp",
        shape=(n_samples,),
        maxshape=(None,),
        dtype=h5py.string_dtype("utf-8"),
    )
    dset.attrs["read_only"] = True
    dset.attrs["num_feature_dims"] = 0
    dset.attrs["units"] = ""
    return dset


def _write_sample(
    datasets: dict[str, h5py.Dataset],
    ts_dset: h5py.Dataset,
    i: int,
    pvs: list[str],
    values: list,
    ts: str,
) -> None:
    if i >= ts_dset.shape[0]:
        ts_dset.resize((i + 1,))
        for d in datasets.values():
            d.resize((i + 1, *d.shape[1:]))

    ts_dset[i] = ts
    for pv, v in zip(pvs, values):
        if pv not in datasets:
            continue
        d = datasets[pv]
        if v is None:
            d[i] = _missing_fill(d)
            continue
        if h5py.check_string_dtype(d.dtype) is not None:
            d[i] = _coerce_string(v)
            continue
        expected_shape = d.shape[1:]
        arr = np.asarray(v)
        if arr.shape != expected_shape:
            logger.warning(
                "PV %s returned shape %s, expected %s; writing NaN",
                pv,
                arr.shape,
                expected_shape,
            )
            d[i] = _nan_block(expected_shape)
        else:
            d[i] = arr


def _initial_read(pvs: list[str], timeout: float) -> list:
    from epics import caget_many

    return caget_many(pvs, timeout=timeout)


def _subsequent_read(pvs: list[str], timeout: float) -> list:
    from epics import caget_many

    return caget_many(pvs, timeout=timeout)


def _run(
    areas: Union[str, Iterable[str]],
    extra_pvs: Iterable[str],
    *,
    path: Union[str, Path],
    n_samples: int,
    rate_hz: float | None,
    duration_s: float | None,
    run_source: str,
    notes: str,
    metadata: dict | None,
    caget_timeout: float,
    include_inactive: bool,
) -> Path:
    path = Path(path)
    pv_loc = _resolve_pvs(areas, extra_pvs, include_inactive)
    if not pv_loc:
        raise ValueError("No PVs resolved for the requested areas / extras.")
    pvs = [p for p, _ in pv_loc]

    first_values = _initial_read(pvs, caget_timeout)
    first_ts = _isotime()

    missing = [pv for pv, v in zip(pvs, first_values) if v is None]
    kept_pv_loc = [(pv, loc) for (pv, loc), v in zip(pv_loc, first_values) if v is not None]
    kept_values = [v for v in first_values if v is not None]

    if not kept_pv_loc:
        raise RuntimeError("All PVs returned None on first read; nothing to archive.")

    batch_id = _new_batch_id(run_source)
    f, batch = _open_file_and_batch(
        path=path,
        batch_id=batch_id,
        n_samples=n_samples,
        source=run_source,
        notes=notes,
        metadata=metadata,
        missing_pvs=missing,
        requested_rate_hz=rate_hz,
        duration_s=duration_s,
    )
    ts_dset: h5py.Dataset | None = None
    datasets: dict[str, h5py.Dataset] = {}
    rows_written = 0
    clean_exit = False
    try:
        obs = batch.create_group("observables")
        ts_dset = _create_timestamp_dataset(obs, n_samples)
        for (pv, loc), v in zip(kept_pv_loc, kept_values):
            datasets[pv] = _create_pv_dataset(obs, pv, loc, n_samples, v)

        kept_pvs = [pv for pv, _ in kept_pv_loc]
        start_mono = time.monotonic()
        deadline_mono = (start_mono + duration_s) if duration_s is not None else None
        _write_sample(datasets, ts_dset, 0, kept_pvs, kept_values, first_ts)
        rows_written = 1

        if n_samples > 1:
            period = 1.0 / rate_hz if rate_hz else 0.0
            target = start_mono + period
            for i in range(1, n_samples):
                if deadline_mono is not None and time.monotonic() >= deadline_mono:
                    break
                sleep_for = target - time.monotonic()
                if deadline_mono is not None:
                    sleep_for = min(sleep_for, deadline_mono - time.monotonic())
                if sleep_for > 0:
                    time.sleep(sleep_for)
                elif target < time.monotonic():
                    logger.warning(
                        "Sample %d slipped by %.3fs", i, time.monotonic() - target
                    )
                if deadline_mono is not None and time.monotonic() >= deadline_mono:
                    break
                values = _subsequent_read(kept_pvs, caget_timeout)
                _write_sample(datasets, ts_dset, i, kept_pvs, values, _isotime())
                rows_written = i + 1
                target += period

        if rate_hz is not None and rows_written > 1:
            elapsed = time.monotonic() - start_mono
            achieved = (rows_written - 1) / elapsed if elapsed > 0 else float("nan")
            notes_payload = json.loads(batch.attrs["run_information_notes"])
            notes_payload["achieved_rate_hz"] = achieved
            batch.attrs["run_information_notes"] = json.dumps(notes_payload)
            if rate_hz and achieved < 0.8 * rate_hz:
                logger.warning(
                    "Achieved %.2f Hz vs requested %.2f Hz over %d PVs; "
                    "caget round-trip is the bottleneck",
                    achieved,
                    rate_hz,
                    len(kept_pvs),
                )
        clean_exit = True
    finally:
        try:
            _truncate_to_written(
                batch, ts_dset, datasets, rows_written, n_samples, clean_exit
            )
        finally:
            f.close()
    return path


def _truncate_to_written(
    batch: h5py.Group,
    ts_dset: h5py.Dataset | None,
    datasets: dict[str, h5py.Dataset],
    rows_written: int,
    requested: int,
    clean_exit: bool,
) -> None:
    """Shrink every dataset to the fully-written row count, update batch_dims.

    ``clean_exit`` distinguishes a normal end (count hit, or duration-mode
    deadline hit before the preallocated row count) from an exception
    bubbling through the try block. Only the latter sets ``interrupted``
    in the notes. No-op when ``ts_dset is None`` (dataset creation never
    ran) or when the preallocated size matches what was written.
    """
    if ts_dset is None or rows_written == requested:
        return
    ts_dset.resize((rows_written,))
    for d in datasets.values():
        d.resize((rows_written, *d.shape[1:]))
    batch.attrs["batch_dims"] = np.array([rows_written], dtype=np.int64)
    notes_payload = json.loads(batch.attrs["run_information_notes"])
    notes_payload["samples_written"] = rows_written
    notes_payload["samples_preallocated"] = requested
    if not clean_exit:
        notes_payload["interrupted"] = True
    batch.attrs["run_information_notes"] = json.dumps(notes_payload)
    if clean_exit:
        logger.info(
            "Archive run ended before preallocated count (likely duration "
            "deadline): wrote %d of %d rows",
            rows_written,
            requested,
        )
    else:
        logger.warning(
            "Archive run interrupted: wrote %d of %d samples; truncated datasets",
            rows_written,
            requested,
        )


def snapshot(
    areas: Union[str, Iterable[str]],
    extra_pvs: Iterable[str] = (),
    *,
    path: Union[str, Path],
    run_source: str = "LCLS",
    notes: str = "",
    metadata: dict | None = None,
    caget_timeout: float = 10.0,
    include_inactive: bool = False,
) -> Path:
    """Read every PV once and write a single-sample Cropp-spec HDF5 file."""
    return _run(
        areas,
        extra_pvs,
        path=path,
        n_samples=1,
        rate_hz=None,
        duration_s=None,
        run_source=run_source,
        notes=notes,
        metadata=metadata,
        caget_timeout=caget_timeout,
        include_inactive=include_inactive,
    )


def continuous(
    areas: Union[str, Iterable[str]],
    extra_pvs: Iterable[str] = (),
    *,
    path: Union[str, Path],
    rate_hz: float = 1.0,
    n_samples: int | None = None,
    duration_s: float | None = None,
    run_source: str = "LCLS",
    notes: str = "",
    metadata: dict | None = None,
    caget_timeout: float = 5.0,
    include_inactive: bool = False,
) -> Path:
    """Sample at ``rate_hz`` and accumulate into one Cropp-spec HDF5 file.

    ``rate_hz`` is a target. Cycle time scales at ~0.15 ms/PV; safe
    sustained rate ~= ``1 / (0.0003 * N_PVs)`` Hz. Achieved rate is
    recorded in ``run_information_notes.achieved_rate_hz``.

    Exactly one of ``n_samples`` or ``duration_s`` must be provided.
    """
    if (n_samples is None) == (duration_s is None):
        raise ValueError("Provide exactly one of n_samples or duration_s.")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive.")
    if n_samples is not None and n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if n_samples is None:
        # Preallocate an upper bound; _run breaks the loop on the real
        # monotonic deadline and _truncate_to_written shrinks to what
        # actually fit. +1 to leave room for the always-written first
        # sample regardless of duration vs cadence rounding.
        n_samples = math.ceil(duration_s * rate_hz) + 1

    return _run(
        areas,
        extra_pvs,
        path=path,
        n_samples=n_samples,
        rate_hz=rate_hz,
        duration_s=duration_s,
        run_source=run_source,
        notes=notes,
        metadata=metadata,
        caget_timeout=caget_timeout,
        include_inactive=include_inactive,
    )
