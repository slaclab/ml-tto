import torch
from unittest.mock import patch

import pytest
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from ml_tto.automatic_emittance.scan_cropping import crop_scan
from ml_tto.automatic_emittance.scan_cropping import crop_scan_by_concavity
from ml_tto.automatic_emittance.scan_cropping import fit_1d_gp_model

import numpy as np


class TestScanCropping:
    @pytest.mark.parametrize("query", [0.0, 0.25])
    def test_rbf_second_derivative(self, query):
        model = fit_1d_gp_model(
            np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])
        )
        kernel = model.covar_module
        fixed_point = torch.zeros(1, 1, dtype=torch.float64)
        query_point = torch.tensor(query, dtype=torch.float64)

        def covariance(point):
            return kernel(point.reshape(1, 1), fixed_point).to_dense().sum()

        actual = torch.autograd.functional.hessian(covariance, query_point)
        lengthscale = kernel.lengthscale.detach().squeeze()
        expected = (
            ((query_point / lengthscale) ** 2 - 1)
            / lengthscale**2
            * torch.exp(-0.5 * (query_point / lengthscale) ** 2)
        )
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("batched", [False, True])
    def test_scan_kernel_preserves_default_gp_predictions(self, batched):
        x = np.array([-1.0, 0.0, 1.0])
        y = np.array([1.0, 0.0, 1.0])
        model = fit_1d_gp_model(x, y)
        reference = SingleTaskGP(
            torch.from_numpy(x).reshape(-1, 1),
            torch.from_numpy(y).reshape(-1, 1),
            input_transform=Normalize(d=1),
            outcome_transform=Standardize(m=1),
        )
        reference.load_state_dict(model.state_dict())
        inputs = torch.linspace(-1.0, 1.0, 11, dtype=torch.float64).reshape(-1, 1)
        if batched:
            inputs = torch.stack([inputs, inputs + 0.25])
        with torch.no_grad():
            posterior = model.posterior(inputs)
            reference_posterior = reference.posterior(inputs)
            torch.testing.assert_close(posterior.mean, reference_posterior.mean)
            torch.testing.assert_close(posterior.variance, reference_posterior.variance)
            for diag in (False, True):
                torch.testing.assert_close(
                    model.covar_module(inputs, inputs, diag=diag).to_dense(),
                    reference.covar_module(inputs, inputs, diag=diag).to_dense(),
                )

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

        # The fit should reproduce the observed squared beam sizes.
        assert np.allclose(
            model.posterior(torch.tensor(x.reshape(-1, 1))).mean.detach().numpy().flatten(),
            y,
            rtol=1e-1,
        )

        # Check training-point curvature independently of kernel autograd.
        step = 1e-3
        inputs = torch.tensor(x.reshape(-1, 1))
        with torch.no_grad():
            numerical_concavity = (
                model.posterior(inputs + step).mean
                - 2 * model.posterior(inputs).mean
                + model.posterior(inputs - step).mean
            ).numpy().flatten() / step**2
        np.testing.assert_allclose(
            concavity_values, numerical_concavity, rtol=1e-3, atol=1e-2
        )
        assert concavity_mask.dtype == bool
        assert np.array_equal(concavity_mask, concavity_values <= 0)
        assert np.array_equal(np.isnan(y_cropped), concavity_mask)
        assert np.array_equal(y_cropped[~concavity_mask], y[~concavity_mask])
        assert concavity_values[0] < 0
        assert concavity_values[-2] < 0
        assert concavity_values[3] > 0  # the middle point should be concave up

    def test_cropping_by_concavity_masks_nonpositive_curvature(self):
        x = np.arange(5.0)
        y = np.array([160000.0, 90000.0, 40000.0, 10000.0, 40000.0])
        original_y = y.copy()
        curvature = np.array([-2.0, 0.0, 3.0, 4.0, -1.0])
        expected_mask = np.array([True, True, False, False, True])

        with (
            patch(
                "ml_tto.automatic_emittance.scan_cropping.fit_1d_gp_model"
            ) as fit_model,
            patch(
                "ml_tto.automatic_emittance.scan_cropping.posterior_mean_concavity",
                return_value=curvature,
            ),
        ):
            cropped, mask, values, model = crop_scan_by_concavity(x, y)

        assert model is fit_model.return_value
        assert np.array_equal(values, curvature)
        assert np.array_equal(mask, expected_mask)
        assert np.array_equal(np.isnan(cropped), expected_mask)
        assert np.array_equal(cropped[~mask], y[~mask])
        assert np.array_equal(y, original_y)
