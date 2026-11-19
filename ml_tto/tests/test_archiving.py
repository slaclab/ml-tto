import json
import sys
import time
import types
from datetime import datetime

import h5py
import numpy as np
import pytest


def _install_fake_slac_db(areas_to_devices, device_to_accessors):
    module = types.ModuleType("slac_db")
    device_mod = types.ModuleType("slac_db.device")

    def get_devices(area=None, include_inactive=False):
        return list(areas_to_devices.get(area, []))

    def get_all_accessors(device):
        return dict(device_to_accessors.get(device, {}))

    device_mod.get_devices = get_devices
    device_mod.get_all_accessors = get_all_accessors
    module.device = device_mod
    sys.modules["slac_db"] = module
    sys.modules["slac_db.device"] = device_mod


def _install_fake_epics(caget_many_fn):
    module = types.ModuleType("epics")
    module.caget_many = caget_many_fn
    sys.modules["epics"] = module


@pytest.fixture
def archiver(request):
    missing = object()
    originals = {
        name: sys.modules.get(name, missing)
        for name in ("slac_db", "slac_db.device", "epics")
    }

    def restore_modules():
        for name, previous in originals.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    request.addfinalizer(restore_modules)

    _install_fake_slac_db(
        areas_to_devices={
            "GUNB": ["SOLN:IN20:121", "QUAD:IN20:121"],
            "HTR": ["QUAD:IN20:121", "QUAD:IN20:361"],
        },
        device_to_accessors={
            "SOLN:IN20:121": {"BACT": "SOLN:IN20:121:BACT", "BCTRL": "SOLN:IN20:121:BCTRL"},
            "QUAD:IN20:121": {"BACT": "QUAD:IN20:121:BACT", "BCTRL": "QUAD:IN20:121:BCTRL"},
            "QUAD:IN20:361": {"BACT": "QUAD:IN20:361:BACT"},
        },
    )
    sys.modules.pop("ml_tto.archiving", None)
    sys.modules.pop("ml_tto.archiving.archiver", None)
    from ml_tto.archiving import archiver as mod

    yield mod


def test_build_pv_list_dedup_and_extras(archiver):
    pv_loc = archiver.build_pv_list(
        ["GUNB", "HTR"],
        extra_pvs=["OTRS:IN20:571:XRMS", "QUAD:IN20:121:BACT"],
    )
    pvs = [p for p, _ in pv_loc]
    locs = dict(pv_loc)

    assert pvs.count("QUAD:IN20:121:BACT") == 1
    assert locs["SOLN:IN20:121:BACT"] == "GUNB"
    assert locs["QUAD:IN20:121:BACT"] == "GUNB"
    assert locs["QUAD:IN20:361:BACT"] == "HTR"
    assert locs["OTRS:IN20:571:XRMS"] == ""
    assert pvs[-1] == "OTRS:IN20:571:XRMS"


def test_snapshot_writes_spec_conformant_h5(archiver, tmp_path):
    image = np.arange(12, dtype=np.float64).reshape(3, 4)

    def fake_caget(pvs, timeout=None):
        return [image if pv.endswith(":IMAGE") else float(i) for i, pv in enumerate(pvs)]

    _install_fake_epics(fake_caget)

    out = tmp_path / "snap.h5"
    archiver.snapshot(
        ["GUNB"],
        extra_pvs=["OTRS:IN20:571:IMAGE"],
        path=out,
        run_source="TEST",
    )

    with h5py.File(out, "r") as f:
        assert f.attrs["Data_Standard_Version"] == "0.1.2"
        ids = [s.decode() if isinstance(s, bytes) else s for s in f.attrs["IDs"]]
        assert len(ids) == 1
        batch = f[ids[0]]
        assert tuple(batch.attrs["batch_dims"]) == (1,)
        assert batch.attrs["run_information_source"] == "TEST"

        obs = batch["observables"]
        ts = obs["Timestamp"]
        assert ts.shape == (1,)
        raw = ts[0]
        ts_str = raw.decode() if isinstance(raw, bytes) else raw
        datetime.fromisoformat(ts_str)
        assert ts.attrs["read_only"]
        assert ts.attrs["num_feature_dims"] == 0

        scalar = obs["SOLN:IN20:121:BACT"]
        assert scalar.shape == (1,)
        assert scalar.attrs["location"] == "GUNB"
        assert scalar.attrs["num_feature_dims"] == 0
        assert scalar.attrs["read_only"]

        img = obs["OTRS:IN20:571:IMAGE"]
        assert img.shape == (1, 3, 4)
        assert img.attrs["location"] == ""
        assert img.attrs["num_feature_dims"] == 2
        assert img.attrs["bin_size"] == 1.0
        assert img.attrs["offset"] == 0.0
        np.testing.assert_array_equal(img[0], image)


def test_snapshot_handles_string_pv(archiver, tmp_path):
    def fake_caget(pvs, timeout=None):
        return [
            "READY" if pv.endswith(":ACCESS") else 1.5
            for pv in pvs
        ]

    _install_fake_epics(fake_caget)

    out = tmp_path / "snap_str.h5"
    archiver.snapshot(
        [],
        extra_pvs=["LGPS:IN20:1:BACT", "LGPS:IN20:1:ACCESS"],
        path=out,
    )

    with h5py.File(out, "r") as f:
        (batch_id,) = [s.decode() if isinstance(s, bytes) else s for s in f.attrs["IDs"]]
        obs = f[batch_id]["observables"]

        s = obs["LGPS:IN20:1:ACCESS"]
        assert h5py.check_string_dtype(s.dtype) is not None
        raw = s[0]
        val = raw.decode() if isinstance(raw, bytes) else raw
        assert val == "READY"

        n = obs["LGPS:IN20:1:BACT"]
        assert n.dtype == np.float64
        assert n[0] == 1.5


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_samples": 0}, "n_samples must be positive"),
        ({"n_samples": -5}, "n_samples must be positive"),
        ({"duration_s": 0}, "duration_s must be positive"),
        ({"duration_s": -0.1}, "duration_s must be positive"),
        ({}, "Provide exactly one"),
        ({"n_samples": 1, "duration_s": 1.0}, "Provide exactly one"),
    ],
)
def test_continuous_rejects_bad_inputs(archiver, tmp_path, kwargs, match):
    _install_fake_epics(lambda pvs, timeout=None: [0.0] * len(pvs))
    with pytest.raises(ValueError, match=match):
        archiver.continuous("GUNB", path=tmp_path / "x.h5", rate_hz=1.0, **kwargs)


def test_continuous_duration_stops_at_deadline(archiver, tmp_path):
    counter = {"n": 0}

    def fake_caget(pvs, timeout=None):
        counter["n"] += 1
        time.sleep(0.12)  # each cycle ~120 ms; rate_hz=50 asks for 20 ms
        return [float(counter["n"] + i) for i, _ in enumerate(pvs)]

    _install_fake_epics(fake_caget)

    out = tmp_path / "duration.h5"
    t0 = time.monotonic()
    archiver.continuous("GUNB", path=out, rate_hz=50.0, duration_s=0.3)
    elapsed = time.monotonic() - t0

    # Should NOT run the ceil(0.3 * 50) + 1 = 16 preallocated iterations
    # (that would take >= 16 * 0.12 = 1.92 s). Must stop near 0.3 s.
    assert elapsed < 1.0, f"duration mode did not stop at deadline: {elapsed:.3f}s"

    with h5py.File(out, "r") as f:
        ids = [s.decode() if isinstance(s, bytes) else s for s in f.attrs["IDs"]]
        batch = f[ids[0]]
        n_written = int(batch.attrs["batch_dims"][0])
        assert 1 <= n_written <= 4  # ~0.3s / 0.12s per cycle
        notes = json.loads(batch.attrs["run_information_notes"])
        assert "interrupted" not in notes  # clean duration exit != interrupted
        assert notes["samples_written"] == n_written
        assert notes["duration_s_requested"] == 0.3

        assert batch["observables"]["Timestamp"].shape == (n_written,)


def test_continuous_truncates_on_interrupt(archiver, tmp_path):
    counter = {"n": 0}

    def fake_caget(pvs, timeout=None):
        counter["n"] += 1
        if counter["n"] >= 4:
            raise KeyboardInterrupt("simulated Ctrl-C")
        return [float(counter["n"] + i) for i, _ in enumerate(pvs)]

    _install_fake_epics(fake_caget)

    out = tmp_path / "interrupted.h5"
    with pytest.raises(KeyboardInterrupt):
        archiver.continuous(
            "GUNB",
            path=out,
            rate_hz=50.0,
            n_samples=10,
        )

    with h5py.File(out, "r") as f:
        ids = [s.decode() if isinstance(s, bytes) else s for s in f.attrs["IDs"]]
        batch = f[ids[0]]
        assert tuple(batch.attrs["batch_dims"]) == (3,)
        notes = json.loads(batch.attrs["run_information_notes"])
        assert notes["interrupted"] is True
        assert notes["samples_written"] == 3
        assert notes["samples_preallocated"] == 10

        obs = batch["observables"]
        assert obs["Timestamp"].shape == (3,)
        for name in obs.keys():
            if name == "Timestamp":
                continue
            assert obs[name].shape[0] == 3


def test_continuous_appends_n_samples(archiver, tmp_path):
    counter = {"n": 0}

    def fake_caget(pvs, timeout=None):
        counter["n"] += 1
        return [float(counter["n"] + i) for i, _ in enumerate(pvs)]

    _install_fake_epics(fake_caget)

    out = tmp_path / "stream.h5"
    archiver.continuous(
        "GUNB",
        path=out,
        rate_hz=50.0,
        n_samples=5,
        metadata={"operator": "test"},
    )

    with h5py.File(out, "r") as f:
        ids = [s.decode() if isinstance(s, bytes) else s for s in f.attrs["IDs"]]
        batch = f[ids[0]]
        assert tuple(batch.attrs["batch_dims"]) == (5,)
        notes = json.loads(batch.attrs["run_information_notes"])
        assert notes["metadata"] == {"operator": "test"}
        assert notes["requested_rate_hz"] == 50.0
        assert "achieved_rate_hz" in notes
        assert notes["achieved_rate_hz"] > 0

        obs = batch["observables"]
        ts = obs["Timestamp"]
        assert ts.shape == (5,)
        strs = [s.decode() if isinstance(s, bytes) else s for s in ts[:]]
        parsed = [datetime.fromisoformat(s) for s in strs]
        assert parsed == sorted(parsed)

        scalar_pvs = [k for k in obs.keys() if k != "Timestamp"]
        assert scalar_pvs, "expected at least one PV dataset"
        for pv in scalar_pvs:
            d = obs[pv]
            assert d.shape == (5,)
            assert np.all(np.diff(d[:]) == 1.0)
