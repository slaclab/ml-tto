from ml_tto.automatic_emittance.screen_profile import (
    ScreenBeamProfileMeasurement,
)
from ml_tto.automatic_emittance.plotting import plot_screen_profile_measurement
from lcls_tools.common.devices.screen import Screen
from unittest.mock import MagicMock
import numpy as np
import matplotlib.pyplot as plt


class TestScreenProfileMeasurement:
    def test_screen_profile_measurement(self):
        # Create a mock Screen device
        screen_device = MagicMock(spec=Screen)
        screen_device.name = "OTRDG02"

        test_image = np.zeros((100, 100))
        test_image[40:60, 40:60] = 1.0
        screen_device.image = test_image

        # Create a ScreenBeamProfileMeasurement instance
        measurement = ScreenBeamProfileMeasurement(beam_profile_device=screen_device)

        # check plotting
        plot_screen_profile_measurement(measurement)
