import unittest
import numpy as np
import matplotlib.pyplot as plt

from ml_tto.automatic_emittance.image_projection_fit import (
    ImageProjectionFit,
    RecursiveImageProjectionFit,
)
from ml_tto.automatic_emittance.plotting import plot_image_projection_fit


class TestImageProjectionFit(unittest.TestCase):
    def setUp(self):
        self.test_image = np.zeros((100, 100))
        self.test_image[40:60, 40:60] = 1.0

    def test_image_projection_fits(self):
        for image_projection_fit in [
            ImageProjectionFit(),
            RecursiveImageProjectionFit(show_intermediate_plots=True),
        ]:
            result = image_projection_fit.fit_image(self.test_image)
            assert np.allclose(result.centroid, [49.5, 49.5])
            assert np.allclose(result.rms_size, [7.7789, 7.7789], rtol=1e-1)
            assert result.total_intensity == 400.0

            image = self.test_image
            x_projection = np.sum(image, axis=0)
            y_projection = np.sum(image, axis=1)

            # calculate fit errors
            x = np.arange(len(x_projection))
            y = np.arange(len(y_projection))
            x_fit_error = np.mean(
                (
                    x_projection
                    - result.projection_fit_method.forward(
                        x, result.x_projection_fit_parameters
                    )
                )
                ** 2
            )
            y_fit_error = np.mean(
                (
                    y_projection
                    - result.projection_fit_method.forward(
                        y, result.y_projection_fit_parameters
                    )
                )
                ** 2
            )

            # calculate noise std
            x_std = np.std(
                x_projection
                - result.projection_fit_method.forward(
                    x, result.x_projection_fit_parameters
                )
            )
            y_std = np.std(
                y_projection
                - result.projection_fit_method.forward(
                    y, result.y_projection_fit_parameters
                )
            )

            assert np.allclose(result.mean_square_errors, [x_fit_error, y_fit_error]), (
                "Fit errors do not match expected values"
            )
            assert np.allclose(result.noise_std, [x_std, y_std]), (
                "Noise std does not match expected values"
            )

            # check the pre-validated fit parameters
            gt_parametetrs = {
                "amplitude": 20.0,
                "mean": 49.5,
                "sigma": 7.7789,
            }
            for key, value in gt_parametetrs.items():
                for i in range(2):
                    assert np.allclose(
                        result.non_validated_parameters[i][key], value, rtol=1e-2
                    ), f"{key} does not match expected value"
                result.x_projection_fit_parameters[key] = value
                result.y_projection_fit_parameters[key] = value

            plot_image_projection_fit(result)
