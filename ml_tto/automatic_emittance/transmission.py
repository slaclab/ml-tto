from lcls_tools.common.measurements.measurement import Measurement
from lcls_tools.common.devices.bpm import BPM
import numpy as np


class TransmissionMeasurement(Measurement):
    name: str = "transmission"
    upstream_bpm: BPM
    downstream_bpm: BPM

    def measure(self):
        """
        Measure the transmission by calculating the ratio of the downstream BPM signal
        to the upstream BPM signal.
        """
        upstream_signal = self.upstream_bpm.tmit
        downstream_signal = self.downstream_bpm.tmit

        if upstream_signal == 0:
            return {"transmission": np.nan}

        transmission = downstream_signal / upstream_signal
        return {"transmission": transmission}
