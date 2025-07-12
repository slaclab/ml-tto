import numpy as np
import pytest
from unittest.mock import create_autospec
from lcls_tools.common.devices.bpm import BPM
from ml_tto.automatic_emittance.transmission import TransmissionMeasurement


class TestTransmissionMeasurement:
    def test_transmission_normal(self):
        upstream = create_autospec(BPM, instance=True)
        downstream = create_autospec(BPM, instance=True)
        upstream.tmit = 10.0
        downstream.tmit = 5.0
        meas = TransmissionMeasurement(upstream_bpm=upstream, downstream_bpm=downstream)
        result = meas.measure()
        assert "transmission" in result
        assert np.isclose(result["transmission"], 0.5)

    def test_transmission_zero_upstream(self):
        upstream = create_autospec(BPM, instance=True)
        downstream = create_autospec(BPM, instance=True)
        upstream.tmit = 0.0
        downstream.tmit = 5.0
        meas = TransmissionMeasurement(upstream_bpm=upstream, downstream_bpm=downstream)
        result = meas.measure()
        assert "transmission" in result
        assert np.isnan(result["transmission"])

    def test_transmission_zero_downstream(self):
        upstream = create_autospec(BPM, instance=True)
        downstream = create_autospec(BPM, instance=True)
        upstream.tmit = 10.0
        downstream.tmit = 0.0
        meas = TransmissionMeasurement(upstream_bpm=upstream, downstream_bpm=downstream)
        result = meas.measure()
        assert "transmission" in result
        assert result["transmission"] == 0.0

    def test_transmission_negative_values(self):
        upstream = create_autospec(BPM, instance=True)
        downstream = create_autospec(BPM, instance=True)
        upstream.tmit = -10.0
        downstream.tmit = -5.0
        meas = TransmissionMeasurement(upstream_bpm=upstream, downstream_bpm=downstream)
        result = meas.measure()
        assert "transmission" in result
        assert np.isclose(result["transmission"], 0.5)

    def test_transmission_upstream_none(self):
        upstream = create_autospec(BPM, instance=True)
        downstream = create_autospec(BPM, instance=True)
        upstream.tmit = None
        downstream.tmit = 5.0
        meas = TransmissionMeasurement(upstream_bpm=upstream, downstream_bpm=downstream)
        with pytest.raises(TypeError):
            meas.measure()
