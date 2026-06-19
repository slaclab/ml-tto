import numpy as np
from typing import Optional

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch import fit_gpytorch_mll
import torch
from gpytorch import ExactMarginalLogLikelihood

import logging

logger = logging.getLogger(__name__)


def _as_1d_float_array(values: np.ndarray) -> np.ndarray:
    """Convert array-like values to a one-dimensional float array.

    Parameters
    ----------
    values : numpy.ndarray
        Input values to normalize.

    Returns
    -------
    numpy.ndarray
        Flattened one-dimensional float array.
    """

    return np.asarray(values, dtype=float).reshape(-1)

def crop_scan_by_concavity(
    scan_values: np.ndarray,
    beam_sizes: np.ndarray,
 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, SingleTaskGP]:
    """Crop points using GP posterior concavity.

    Fit a 1D GP model to the square of the beam sizes as a function of the scan values, 
    then identify which scan points are in regions of upward concavity based
    on the second derivative of the GP posterior mean. Points where the concavity is not
    upward are masked out (set to NaN) in the returned beam size array.

    Parameters
    ----------
    scan_values : numpy.ndarray
        One-dimensional scan values in machine units.
    beam_sizes : numpy.ndarray
        One-dimensional beam sizes in meters.

    Returns
    -------
    tuple
        (beam_sizes_cropped, concavity_mask, concavity_values, model).
    """

    scan_values = _as_1d_float_array(scan_values)
    beam_sizes = _as_1d_float_array(beam_sizes)

    scan_values_no_nans = scan_values[~np.isnan(beam_sizes)]
    beam_sizes_no_nans = beam_sizes[~np.isnan(beam_sizes)]
    beam_sizes_cropped = np.copy(beam_sizes)

    # fit 1d gp model to data
    model = fit_1d_gp_model(scan_values_no_nans, beam_sizes_no_nans**2)

    # identify which scan points are in regions of model upwards concavity
    concavity_values = posterior_mean_concavity(model, scan_values)
    data_is_concave_up = concavity_values > 0

    # set beam size data to nan where concavity is not upward
    concavity_mask = ~data_is_concave_up
    beam_sizes_cropped[concavity_mask] = np.nan

    return beam_sizes_cropped, concavity_mask, concavity_values, model


def crop_scan_by_beam_size(
    beam_sizes: np.ndarray,
    cutoff_max: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop points where beam size exceeds a threshold.

    Parameters
    ----------
    beam_sizes : numpy.ndarray
        One-dimensional beam sizes in meters.
    cutoff_max : float or None
        Upper beam-size threshold in meters. If None, no cutoff is applied.

    Returns
    -------
    tuple
        (beam_sizes_cropped, cutoff_mask).
    """

    beam_sizes_cropped = _as_1d_float_array(beam_sizes)
    if cutoff_max is not None:
        cutoff_mask = beam_sizes_cropped > cutoff_max
    else:
        cutoff_mask = np.zeros(len(beam_sizes_cropped), dtype=bool)
    beam_sizes_cropped[cutoff_mask] = np.nan

    return beam_sizes_cropped, cutoff_mask


def crop_scan(
    scan_values: np.ndarray,
    beam_sizes: np.ndarray,
    cutoff_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[SingleTaskGP]]:
    """Apply beam-size cutoff and concavity cropping to a scan.

    Parameters
    ----------
    scan_values : numpy.ndarray
        One-dimensional scan values in machine units.
    beam_sizes : numpy.ndarray
        One-dimensional beam sizes in meters.
    cutoff_max : float or None, optional
        Upper beam-size threshold in meters.
    Returns
    -------
    tuple
        (scan_values_cropped, beam_sizes_cropped, concavity_mask, cutoff_mask, model).
        The scan values are returned unchanged for alignment with the cropped
        beam-size array. If GP fitting fails, model is None and concavity_mask
        is all False while beam-size cutoff cropping is preserved.
    """

    scan_values_array = np.asarray(scan_values, dtype=float)
    beam_sizes_array = np.asarray(beam_sizes, dtype=float)
    beam_sizes_shape = beam_sizes_array.shape

    scan_values = _as_1d_float_array(scan_values_array)
    beam_sizes = _as_1d_float_array(beam_sizes_array)

    # Ensure both arrays stay aligned before applying masks.
    npts = min(len(scan_values), len(beam_sizes))
    scan_values = scan_values[:npts]
    beam_sizes = beam_sizes[:npts]

    # copy input data before making edits
    scan_values_cropped = np.copy(scan_values)

    # Always apply the direct beam-size cutoff first.
    beam_sizes_after_cutoff, cutoff_mask = crop_scan_by_beam_size(
        beam_sizes=beam_sizes,
        cutoff_max=cutoff_max,
    )

    beam_sizes_cropped, concavity_mask, concavity_values, model = crop_scan_by_concavity(
        scan_values=scan_values,
        beam_sizes=beam_sizes_after_cutoff,
    )

    # Keep output masks and beam sizes aligned with input beam-size shape.
    if beam_sizes_shape and np.prod(beam_sizes_shape) == len(beam_sizes_cropped):
        beam_sizes_cropped = beam_sizes_cropped.reshape(beam_sizes_shape)
        concavity_mask = concavity_mask.reshape(beam_sizes_shape)
        cutoff_mask = cutoff_mask.reshape(beam_sizes_shape)

    return (
        scan_values_cropped,
        beam_sizes_cropped,
        concavity_mask,
        cutoff_mask,
        concavity_values,
        model,
    )


def fit_1d_gp_model(x: np.ndarray, y: np.ndarray) -> SingleTaskGP:
    """Fit a one-dimensional GP regression model.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional training inputs.
    y : numpy.ndarray
        One-dimensional training targets. Must not contain NaNs.

    Returns
    -------
    SingleTaskGP
        Trained GP model for the provided data.
    """

    # fit a 1d GP model to the scan data
    train_X = torch.from_numpy(x).reshape(-1, 1)
    train_Y = torch.from_numpy(y).reshape(-1, 1)
    outcome_transform = Standardize(m=1)
    input_transform = Normalize(d=1)
    model = SingleTaskGP(
        train_X,
        train_Y,
        outcome_transform=outcome_transform,
        input_transform=input_transform,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


def posterior_mean_second_derivative(
    model: SingleTaskGP, x_values: torch.tensor
) -> torch.tensor:
    """Evaluate the second derivative of the GP posterior mean.

    Parameters
    ----------
    model : SingleTaskGP
        Trained one-dimensional GP model.
    x_values : torch.Tensor
        Points at which the second derivative is evaluated.

    Returns
    -------
    torch.Tensor
        Second derivative values of the posterior mean at x_values.
    """

    x_values = x_values.reshape(-1)

    def posterior_mean_sum(x_values):
        return model.posterior(x_values.reshape(-1, 1)).mean.sum()

    d2y_dx2 = torch.diag(
        torch.autograd.functional.hessian(posterior_mean_sum, x_values)
    )

    return d2y_dx2


def posterior_mean_concavity(
    model: SingleTaskGP, x_values: np.ndarray, visualize: bool = False
) -> np.ndarray:
    """Compute the posterior mean concavity of a GP model at given points.

    Parameters
    ----------
    model : SingleTaskGP
        Trained one-dimensional GP model.
    x_values : numpy.ndarray
        One-dimensional array of points for concavity evaluation.
    visualize : bool, optional
        Unused compatibility argument.

    Returns
    -------
    numpy.ndarray
        Array of second derivative values of the posterior mean at x_values.
    """

    x_values = torch.from_numpy(_as_1d_float_array(x_values))
    d2y_dx2 = posterior_mean_second_derivative(model, x_values)

    return d2y_dx2.detach().numpy()
