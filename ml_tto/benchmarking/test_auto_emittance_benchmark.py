from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import torch
from cheetah import Segment, Quadrupole, Drift, ParameterBeam

from lcls_tools.common.devices.magnet import Magnet, MagnetMetadata
from lcls_tools.common.devices.reader import create_magnet
from lcls_tools.common.devices.screen import Screen
from lcls_tools.common.frontend.plotting.emittance import plot_quad_scan_result
from lcls_tools.common.image.roi import CircularROI
from lcls_tools.common.image.processing import ImageProcessor

from ml_tto.automatic_emittance.automatic_emittance import (
    MLQuadScanEmittance,
)
from ml_tto.automatic_emittance.screen_profile import (
    ScreenBeamProfileMeasurement,
    ScreenBeamProfileMeasurementResult,
)


class MockBeamline:
    def __init__(self, initial_beam: ParameterBeam):
        """create mock beamline, powered by cheetah"""
        self.beamline = Segment(
            [
                Quadrupole(name="Q0", length=torch.tensor(0.1)),
                Drift(length=torch.tensor(1.0)),
            ]
        )

        self.magnet = MagicMock(spec=Magnet)

        # add a property to the magnet to control the quad strength
        type(self.magnet).bctrl = property(self.get_bctrl, self.set_bctrl)
        type(self.magnet).bact = property(self.get_bact)

        self.magnet.metadata = MagnetMetadata(
            area="test", beam_path=["test"], sum_l_meters=None, l_eff=0.1
        )

        self.roi = CircularROI(center=[1, 1], radius=1000)
        self.screen_resolution = 1.0  # resolution of the screen in um / px
        self.beamsize_measurement = MagicMock(spec=ScreenBeamProfileMeasurement)
        self.beamsize_measurement.device = MagicMock(spec=Screen)
        self.beamsize_measurement.device.resolution = self.screen_resolution
        self.beamsize_measurement.image_processor = MagicMock()
        self.beamsize_measurement.image_processor.roi = self.roi
        self.beamsize_measurement.measure = MagicMock(
            side_effect=self.get_beamsize_measurement
        )

        self.initial_beam = initial_beam

    def get_bctrl(self, *args):
        return self.beamline.Q0.k1.numpy()

    def get_bact(self, *args):
        return self.beamline.Q0.k1.numpy()

    def set_bctrl(self, *args):
        # NOTE: this is a bit of a hack since the first argument is the MagicMock object
        self.beamline.Q0.k1 = torch.tensor(args[1])

    def get_beamsize_measurement(self, *args):
        """define a mock beamsize measurement for the
        ScreenBeamProfileMeasurement -- returns image fit result in pixels"""
        outgoing_beam = self.beamline.track(self.initial_beam)

        sigma_x = (
            outgoing_beam.sigma_x * 1e6 / self.screen_resolution
            + 5.0 * np.random.randn(args[0])
        )
        sigma_y = (
            outgoing_beam.sigma_y * 1e6 / self.screen_resolution
            + 5.0 * np.random.randn(args[0])
        )

        result = MagicMock(ScreenBeamProfileMeasurementResult)
        result.rms_sizes = np.stack([sigma_x, sigma_y]).T
        result.centroids = self.roi.radius[0] * np.ones((args[0], 2))
        result.signal_to_noise_ratios = np.ones(2) * 10.0

        # simulate the beam losing intensity on the edges
        intensity = 10 ** (6.0 - 0.5 * np.abs(self.beamline.Q0.k1.numpy()))
        # intensity = 1e6

        result.total_intensities = np.ones(args[0]) * intensity
        result.metadata = MagicMock()
        result.metadata.image_processor = MagicMock()
        result.metadata.image_processor.roi = self.roi

        return result


def run_calc():
    rmat = np.array([[[1, 1.0], [0, 1]], [[1, 1.0], [0, 1]]])

    initial_beam = ParameterBeam.from_twiss(
        beta_x=torch.tensor(5.0) + torch.randn(1) * 0.1,
        alpha_x=torch.tensor(5.0) + torch.randn(1) * 0.1,
        emittance_x=torch.tensor(1e-8),
        beta_y=torch.tensor(3.0) + torch.randn(1) * 0.1,
        alpha_y=torch.tensor(3.0) + torch.randn(1) * 0.1,
        emittance_y=torch.tensor(1e-7),
    )

    mock_beamline = MockBeamline(initial_beam)

    # Instantiate the QuadScanEmittance object
    quad_scan = MLQuadScanEmittance(
        energy=1e9 * 299.792458 / 1e3,
        magnet=mock_beamline.magnet,
        beamsize_measurement=mock_beamline.beamsize_measurement,
        n_measurement_shots=3,
        rmat=rmat,
        n_initial_samples=3,
        n_iterations=8,
        max_scan_range=[-10, 10],
    )

    # Call the measure method
    result = quad_scan.measure()

    # check resulting calculations against cheetah simulation ground truth
    assert np.allclose(
        result.emittance,
        np.array([1.0e-2, 1.0e-1]).reshape(2, 1),
        rtol=1.0e-1,
    )

    # check the reconstructed beam matrix parameters
    assert np.allclose(
        result.beam_matrix[0],
        np.array(
            [
                initial_beam._cov[0, 0, 0].numpy(),
                initial_beam._cov[0, 0, 1].numpy(),
                initial_beam._cov[0, 1, 1].numpy(),
            ]
        )
        * 1e6,
        rtol=1.0e-1,
    )


@pytest.mark.benchmark(max_time=30.0)
def test_robustness(benchmark):
    benchmark(run_calc)
