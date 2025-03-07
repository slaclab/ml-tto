from unittest.mock import patch, Mock, MagicMock

import numpy as np
import torch
from cheetah import Segment, Quadrupole, Drift, ParameterBeam

from lcls_tools.common.devices.magnet import Magnet, MagnetMetadata
from lcls_tools.common.devices.screen import Screen
from lcls_tools.common.frontend.plotting.emittance import plot_quad_scan_result
from lcls_tools.common.image.roi import CircularROI
from lcls_tools.common.image.processing import ImageProcessor

from ml_tto.automatic_emittance.automatic_emittance import (
    MLQuadScanEmittance,
)
from ml_tto.saver import H5Saver
from ml_tto.automatic_emittance.screen_profile import ScreenBeamProfileMeasurement, ScreenBeamProfileMeasurementResult



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


class TestAutomaticEmittance:
    def test_evaluate_function(self):
        initial_beam = ParameterBeam.from_twiss(
            beta_x=torch.tensor(5.0),
            alpha_x=torch.tensor(5.0),
            emittance_x=torch.tensor(1e-8),
            beta_y=torch.tensor(3.0),
            alpha_y=torch.tensor(3.0),
            emittance_y=torch.tensor(1e-7),
        )

        mock_beamline = MockBeamline(initial_beam)

        # Instantiate the QuadScanEmittance object
        quad_scan = MLQuadScanEmittance(
            energy=1e9 * 299.792458 / 1e3,
            magnet=mock_beamline.magnet,
            beamsize_measurement=mock_beamline.beamsize_measurement,
            n_measurement_shots=1,
            wait_time=1e-3,
            n_initial_samples=1,
            n_iterations=1,
            max_scan_range=[-10, 10],
        )

        # Call the evaluate method
        inputs = {"k": 1.0}
        result = quad_scan._evaluate(inputs)
        assert isinstance(result, dict)
        # assert "bb_penalty" in result
        # assert "log10_total_intensity" in result
        assert "scaled_x_rms_px" in result
        assert "scaled_y_rms_px" in result

        # test case where k > 5.0 -- should return low intensity and NaN beam sizes
        # inputs = {"k": 10.0}
        # result = quad_scan._evaluate(inputs)
        # assert np.all(np.isnan(result["scaled_x_rms_px"]))
        # assert np.all(np.isnan(result["scaled_y_rms_px"]))
        # assert np.all(np.isnan(result["bb_penalty"]))

        # assert len(quad_scan._info) == 2
        # assert np.all(np.isnan(quad_scan._info[-1].rms_sizes))
        # assert np.all(np.isnan(quad_scan._info[-1].centroids))

    def test_automatic_emit_scan_with_mocked_beamsize_measurement(self):
        """
        Test to verify correct emittance calculation based on data generated from a
        basic cheetah simulation of a quad and drift element
        """
        rmat = np.array([[[1, 1.0], [0, 1]], [[1, 1.0], [0, 1]]])
        design_twiss = {
            "beta_x": 0.2452,
            "alpha_x": -0.1726,
            "beta_y": 0.5323,
            "alpha_y": -1.0615,
        }

        # run test with and without design_twiss
        for design_twiss_ele in [None, design_twiss]:
            for n_shots in [1, 3]:
                initial_beam = ParameterBeam.from_twiss(
                    beta_x=torch.tensor(5.0),
                    alpha_x=torch.tensor(5.0),
                    emittance_x=torch.tensor(1e-8),
                    beta_y=torch.tensor(3.0),
                    alpha_y=torch.tensor(3.0),
                    emittance_y=torch.tensor(1e-7),
                )

                mock_beamline = MockBeamline(initial_beam)
                mock_beamline.magnet.bctrl = 0.01

                # Instantiate the QuadScanEmittance object
                quad_scan = MLQuadScanEmittance(
                    energy=1e9 * 299.792458 / 1e3,
                    magnet=mock_beamline.magnet,
                    beamsize_measurement=mock_beamline.beamsize_measurement,
                    n_measurement_shots=n_shots,
                    wait_time=1e-3,
                    rmat=rmat,
                    design_twiss=design_twiss_ele,
                    n_initial_samples=3,
                    n_iterations=3,
                    max_scan_range=[-10, 10],
                )

                # Call the measure method
                result = quad_scan.measure()

                plot_quad_scan_result(result)

                quad_scan.X.generator.visualize_model(
                    exponentiate=True,
                    show_feasibility=True,
                )
                quad_scan.X.data.plot(y="k")

                # check resulting calculations against cheetah simulation ground truth
                assert np.allclose(
                    result.emittance,
                    np.array([1.0e-2, 1.0e-1]).reshape(2, 1),
                    rtol=1.5e-1,
                )
                assert np.allclose(
                    result.beam_matrix,
                    np.array([[5.0e-2, -5.0e-2, 5.2e-2], [0.3, -0.3, 0.33333328]]),
                    rtol=2.0e-1,
                )

                # make sure that we return the initial quadrupole setting at the end
                assert mock_beamline.magnet.bctrl == 0.01
                
    def test_file_dump(self):
        initial_beam = ParameterBeam.from_twiss(
            beta_x=torch.tensor(5.0),
            alpha_x=torch.tensor(5.0),
            emittance_x=torch.tensor(1e-8),
            beta_y=torch.tensor(3.0),
            alpha_y=torch.tensor(3.0),
            emittance_y=torch.tensor(1e-7),
        )

        mock_beamline = MockBeamline(initial_beam)

        rmat = np.array([[[1, 1.0], [0, 1]], [[1, 1.0], [0, 1]]])
        design_twiss = {
            "beta_x": 0.2452,
            "alpha_x": -0.1726,
            "beta_y": 0.5323,
            "alpha_y": -1.0615,
        }

        screen = MagicMock(Screen)

        # create a mock Screen device
        def mock_get_image(*args):
            image = np.zeros((100, 100))
            image[40:60, 40:60] = 255
            return image

        type(screen).image = property(mock_get_image)
        screen.resolution = 1.0

        image_processor = ImageProcessor(roi=CircularROI(center=[50, 50], radius=50))
        screen_measurement = ScreenBeamProfileMeasurement(
            device=screen,
            image_processor=image_processor,
        )

        # Instantiate the QuadScanEmittance object
        quad_scan = MLQuadScanEmittance(
            energy=1e9 * 299.792458 / 1e3,
            magnet=mock_beamline.magnet,
            beamsize_measurement=screen_measurement,
            n_measurement_shots=3,
            wait_time=1e-3,
            rmat=rmat,
            design_twiss=design_twiss,
            n_initial_samples=1,
            n_iterations=1,
            max_scan_range=[-10, 10],
            save_location=".",
        )

        # Call the measure method
        result = quad_scan.measure()

        # Save results to file
        result_dict = result.model_dump()
        saver = H5Saver()
        saver.dump(result_dict, "emittance_test.h5")

        # Load results from file
        loaded_dict = saver.load("emittance_test.h5")

        # Check if the loaded dictionary is the same as the original
        assert result_dict.keys() == loaded_dict.keys()
        # TODO: continue test
