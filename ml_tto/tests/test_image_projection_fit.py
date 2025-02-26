import unittest
import numpy as np
import matplotlib.pyplot as plt
import pytest

from ml_tto.automatic_emittance.image_projection_fit import (
    ImageProjectionFit,
    RecursiveImageProjectionFit,
)
from ml_tto.automatic_emittance.plotting import plot_image_projection_fit


class TestImageProjectionFit:
    @pytest.mark.parametrize("image_projection_fit", [ImageProjectionFit(), RecursiveImageProjectionFit()])
    def test_image_projection_fits(self, image_projection_fit):
        test_image = np.zeros((100, 100))
        test_image[40:60, 40:60] = 1.0


        result = image_projection_fit.fit_image(test_image)
        assert np.allclose(result.centroid, [49.5, 49.5])
        assert np.allclose(result.rms_size, [7.7789, 7.7789], rtol=1e-1)
        assert result.total_intensity == 400.0

        param_keys = {"amplitude", "mean", "sigma", "offset"}
        for i in range(2):
            fit_params = result.projection_fit_parameters[i]
            # test to make sure the correct keys are returned
            assert set(fit_params.keys()) == param_keys
            assert np.allclose(
                result.signal_to_noise_ratio[i],
                fit_params["amplitude"] / result.noise_std[i],
            ), "Signal to noise ratio does not match expected value"

            assert np.allclose(fit_params["sigma"], 7.7789, rtol=1e-2)

        plot_image_projection_fit(result)

    @pytest.mark.parametrize("image_projection_fit", [ImageProjectionFit(), RecursiveImageProjectionFit()])
    def test_image_projection_fit_with_bad_data(self, image_projection_fit):

        # test case where both directions are bad
        image = np.zeros((100, 100))
        image[10:90, 10:90] = 1.0
        
        result = image_projection_fit.fit_image(image)
        assert np.all(np.isnan(result.centroid))
        assert np.all(np.isnan(result.rms_size))

        for ele in result.projection_fit_parameters:
            for name,val in ele.items():
                assert np.all(np.isnan(val)), f"{name} is not NaN"

        # test case where one direction is bad
        image = np.zeros((100, 100))
        image[10:90, 40:60] = 1.0

        result = image_projection_fit.fit_image(image)
        assert np.isnan(result.centroid[1])
        assert np.isnan(result.rms_size[1])
        assert np.allclose(result.centroid[0], 49.5, rtol=1e-2)
        assert np.allclose(result.rms_size[0], 7.7789, rtol=1e-2)

        # check fit parameters
        assert np.all(np.isnan(np.array(list(result.projection_fit_parameters[1].values())).astype(np.float64)))
        assert np.allclose(result.projection_fit_parameters[0]["sigma"], 7.7789, rtol=1e-2)
        
    
            
