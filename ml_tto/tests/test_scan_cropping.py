import torch

from ml_tto.automatic_emittance.scan_cropping import crop_scan, crop_scan_by_beam_size
from ml_tto.automatic_emittance.scan_cropping import crop_scan_by_concavity

import numpy as np


class TestScanCropping:
    def test_cropping_return_details(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = (np.array([3.0e-4, 2.0e-4, 1.0e-4, 2.0e-4, 3.0e-4])*1e6)**2

        (
            x_cropped,
            y_cropped,
            concavity_mask,
            cutoff_mask,
            concavity_values,
            model,
        ) = crop_scan(x, y, cutoff_max=1000)

        assert x_cropped.shape == x.shape
        assert y_cropped.shape == y.shape
        assert concavity_mask.shape == y.shape
        assert cutoff_mask.shape == y.shape
        assert concavity_values.shape == y.shape
        assert concavity_mask.dtype == bool
        assert cutoff_mask.dtype == bool
        assert model is not None

    def test_cropping_by_concavity_returns_masks_and_model(self):

        x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        y = (np.array([4.0e-4, 3.0e-4, 2.0e-4, 1.0e-4, 2.0e-4, 3.0e-4, 2.0e-4])*1e6)**2

        y_cropped, concavity_mask, concavity_values, model = crop_scan_by_concavity(x, y)

        # validate that the posterior mean GP agrees with the square of the original data at the observed points
        assert np.allclose(
            model.posterior(torch.tensor(x.reshape(-1, 1), dtype=torch.float32)).mean.detach().numpy().flatten(),
            y,
            rtol=1e-1,
        )

        # check the returned values
        assert np.isnan(y_cropped[0])  # the first point should be cropped
        assert np.isnan(y_cropped[-2])  # the second to last point should be cropped

        # assert that the concavity mask correctly identifies the first and last points as not concave down
        assert concavity_mask[0] == True
        assert concavity_mask[-2] == True

        # assert that the concavity values are negative for the points that are concave down
        assert concavity_values[0] < 0
        assert concavity_values[-2] < 0
        assert concavity_values[3] > 0  # the middle point should be concave up

