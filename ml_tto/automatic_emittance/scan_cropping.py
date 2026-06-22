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


def crop_scan_by_concavity(
    scan_values: np.ndarray,
    beam_sizes_squared: np.ndarray,
 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, SingleTaskGP]:
    """Crop points using GP posterior concavity.

    Fit a 1D GP model to the beam sizes as a function of the scan values, 
    then identify which scan points are in regions of upward concavity based
    on the second derivative of the GP posterior mean. Points where the concavity is not
    upward are masked out (set to NaN) in the returned beam size array.

    Parameters
    ----------
    scan_values : numpy.ndarray
        One-dimensional scan values in machine units.
    beam_sizes_squared : numpy.ndarray
        One-dimensional squared beam sizes in um^2.

    Returns
    -------
    tuple
        (beam_sizes_cropped, concavity_mask, concavity_values, model).
    """


    scan_values_no_nans = scan_values[~np.isnan(beam_sizes_squared)]
    beam_sizes_squared_no_nans = beam_sizes_squared[~np.isnan(beam_sizes_squared)]

    # fit 1d gp model to data
    model = fit_1d_gp_model(scan_values_no_nans, beam_sizes_squared_no_nans)

    # identify which scan points are in regions of model upwards concavity
    concavity_values = posterior_mean_concavity(model, scan_values)
    data_is_concave_up = concavity_values > 0

    # set beam size data to nan where concavity is not upward
    concavity_mask = ~data_is_concave_up
    beam_sizes_squared_cropped = np.copy(beam_sizes_squared)
    beam_sizes_squared_cropped[concavity_mask] = np.nan

    return beam_sizes_squared_cropped, concavity_mask, concavity_values, model


def crop_scan_by_beam_size(
    beam_sizes: np.ndarray,
    cutoff_min: Optional[float],
    cutoff_max: Optional[float],
) -> np.ndarray:
    """Crop points where beam size exceeds a threshold.

    Parameters
    ----------
    beam_sizes : numpy.ndarray
        One-dimensional beam sizes in microns.
    cutoff_min : float or None
        Lower beam-size threshold in microns. If None, no cutoff is applied.
    cutoff_max : float or None
        Upper beam-size threshold in microns. If None, no cutoff is applied.

    Returns
    -------
    numpy.ndarray
        Boolean mask indicating which points exceed the beam-size thresholds.
    """

    beam_sizes_cropped = np.copy(beam_sizes)

    cutoff_mask = np.zeros_like(beam_sizes_cropped, dtype=bool)
    if cutoff_max is not None:
        cutoff_mask |= beam_sizes_cropped > cutoff_max
    
    if cutoff_min is not None:
        cutoff_mask |= beam_sizes_cropped < cutoff_min
    
    return cutoff_mask


def crop_scan(
    scan_values: np.ndarray,
    beam_sizes_squared: np.ndarray,
    cutoff_min: Optional[float] = None,
    cutoff_max: Optional[float] = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[SingleTaskGP],
]:

    Parameters
    ----------
    scan_values : numpy.ndarray
        One-dimensional scan values in machine units.
    beam_sizes_squared : numpy.ndarray
        One-dimensional squared beam sizes in microns squared.
    cutoff_min : float or None, optional
        Lower beam-size threshold in microns. If None, no cutoff is applied.
    cutoff_max : float or None, optional
        Upper beam-size threshold in microns. If None, no cutoff is applied.
    Returns
    -------
    tuple
        (scan_values_cropped, beam_sizes_cropped, concavity_mask, cutoff_mask, model).
        The scan values are returned unchanged for alignment with the cropped
        beam-size array. If GP fitting fails, model is None and concavity_mask
        is all False while beam-size cutoff cropping is preserved.
    """
    assert scan_values.ndim == 1, "scan_values must be a one-dimensional array"
    assert beam_sizes_squared.ndim == 1, "beam_sizes_squared must be a one-dimensional array"
    assert scan_values.shape == beam_sizes_squared.shape, "scan_values and beam_sizes_squared must have the same shape"

    cropping_mask = np.zeros_like(beam_sizes_squared, dtype=bool)

    # Always apply the direct beam-size cutoff first -- note that the cutoff is applied to the square root of the input beam sizes squared, which are in microns.
    cutoff_mask = crop_scan_by_beam_size(
        beam_sizes= np.sqrt(beam_sizes_squared),
        cutoff_min=cutoff_min,
        cutoff_max=cutoff_max,
    )
    cropping_mask |= cutoff_mask

    masked_beam_sizes_squared = np.copy(beam_sizes_squared)
    masked_beam_sizes_squared[cropping_mask] = np.nan

    _, concavity_mask, concavity_values, model = crop_scan_by_concavity(
        scan_values=scan_values,
        beam_sizes_squared=masked_beam_sizes_squared,
    )
    cropping_mask |= concavity_mask

    masked_beam_sizes_squared[cropping_mask] = np.nan

    return (
        scan_values,
        masked_beam_sizes_squared,
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

    x_values = torch.from_numpy(x_values)
    d2y_dx2 = posterior_mean_second_derivative(model, x_values)

    return d2y_dx2.detach().numpy()
