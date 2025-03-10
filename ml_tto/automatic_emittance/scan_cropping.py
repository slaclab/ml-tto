import numpy as np
from matplotlib import pyplot as plt
from typing import Optional

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch import fit_gpytorch_mll
import torch
from gpytorch import ExactMarginalLogLikelihood


def crop_scan(
    scan_values: np.ndarray,
    beam_sizes: np.ndarray,
    cutoff_max: Optional[float] = None,
    visualize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replaces beam_sizes greater than
    cutoff_max * minimum observed beam_size with nan.
    Finds regions of upward concavity in remaining data using
    1d GP regression (posterior mean) and replaces beam_sizes
    outside of these regions with nan.
    Plots cropping results if specified.

    Inputs:
        scan_values: 1d numpy array of scan input values
        beam_sizes: 1d numpy array of scan output values
                    (same length as scan_values)
        cutoff_max: float specifying upper limit at which to crop
                    beam_sizes.
                    (Upper limit = cutoff_max * minimum beam_size)
        visualize: boolean specifying whether to plot cropping results
    Outputs:
        scan_values_cropped: 1d numpy array of cropped scan_values
        beam_sizes_cropped: 1d numpy array of cropped beam_sizes
    """

    # remove nans and copy input data before making edits
    scan_values = scan_values[~np.isnan(beam_sizes)]
    beam_sizes = beam_sizes[~np.isnan(beam_sizes)]
    scan_values_cropped = np.copy(scan_values)
    beam_sizes_cropped = np.copy(beam_sizes)

    # fit 1d gp model to data
    model = fit_1d_gp_model(scan_values, beam_sizes**2)

    # identify which scan points are in regions of model upwards concavity
    data_is_concave_up = posterior_mean_concavity(model, scan_values)

    # set beam size data to nan where concavity is not upward
    concavity_mask = ~data_is_concave_up
    beam_sizes_cropped[concavity_mask] = np.nan

    # set beam size data to nan where the beam size is larger than some amount
    if cutoff_max is not None:
        min_observed_size = np.nanmin(beam_sizes)
        cutoff_mask = beam_sizes_cropped > cutoff_max * min_observed_size
    else:
        cutoff_mask = np.zeros(len(beam_sizes_cropped), dtype=bool)
    beam_sizes_cropped[cutoff_mask] = np.nan

    # remove nans again
    scan_values_cropped = scan_values_cropped[~np.isnan(beam_sizes_cropped)]
    beam_sizes_cropped = beam_sizes_cropped[~np.isnan(beam_sizes_cropped)]

    if visualize:
        # evaluate the GP fit and its concavity on a linspace
        fit_x = torch.linspace(
            scan_values.min(), scan_values.max(), 100
        ).reshape(-1, 1)
        fit_y = model.posterior(fit_x).mean.detach().numpy().flatten()
        fit_x = fit_x.detach().numpy().flatten()
        fit_is_concave_up = posterior_mean_concavity(model, fit_x)

        plt.figure()
        if cutoff_max is not None:
            # plot the beam size cutoff_max boundary
            plt.axhline(
                (cutoff_max * min_observed_size) ** 2,
                ls=":",
                c="k",
                label="cutoff",
                zorder=2,
            )
        fit_y_up = np.ma.masked_array(fit_y, mask=~fit_is_concave_up)
        fit_y_down = np.ma.masked_array(fit_y, mask=fit_is_concave_up)
        # plot the GP posterior mean where the concavity it upward
        plt.plot(
            fit_x,
            fit_y_up,
            ls="--",
            c="C1",
            label="concave up",
            zorder=1,
        )
        # plot the GP posterior mean where the concavity is downward
        plt.plot(
            fit_x,
            fit_y_down,
            ls="--",
            c="C2",
            label="concave down",
            zorder=1,
        )
        # plot the data that has been removed by the cutoff_max condition
        plt.scatter(
            scan_values[cutoff_mask],
            beam_sizes[cutoff_mask] ** 2,
            s=50,
            facecolors="none",
            edgecolors="C0",
            label="data removed",
            zorder=3,
        )
        # plot the data that has been removed by the concavity condition
        plt.scatter(
            scan_values[~cutoff_mask * concavity_mask],
            beam_sizes[~cutoff_mask * concavity_mask] ** 2,
            s=50,
            facecolors="none",
            edgecolors="C0",
            zorder=3,
        )
        # plot the data that has remains after cropping
        plt.scatter(
            scan_values_cropped,
            beam_sizes_cropped**2,
            s=50,
            marker="+",
            c="C0",
            label="data retained",
            zorder=3,
        )
        plt.legend()
        plt.ylabel("$Beam size^2 (m^2)$")
        plt.xlabel("Quad value (machine units)")
        plt.title("Inflection point crop")
        plt.tight_layout()

    return scan_values_cropped, beam_sizes_cropped


def fit_1d_gp_model(x: np.ndarray, y: np.ndarray) -> SingleTaskGP:
    """
    Fits 1d GP regression model to 1d training data.

    NOTE: y must not contain NaNs

    Inputs:
        x: 1d numpy array containing training inputs
        y: 1d numpy array containing corresponding training outputs
    Outputs:
        model: SingleTaskGP regression model fit to 1d training data (x, y)
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
    """
    Evaluate the second derivative of the model (GP) posterior mean
    at the given x-values with respect to x.

    Inputs:
        model: SingleTaskGP regression model trained on 1d data
        x_values: 1d torch tensor specifying the inputs at which
                    to evaluate second derivative
    Outputs:
        d2y_dx2: 1d torch tensor containing second derivates of GP
                posterior mean at the given x-values
    """

    def posterior_mean_sum(x_values):
        return model.posterior(x_values.reshape(-1, 1)).mean.sum()

    d2y_dx2 = torch.diag(
        torch.autograd.functional.hessian(posterior_mean_sum, x_values)
    )

    return d2y_dx2


def posterior_mean_concavity(
    model: SingleTaskGP, x_values: np.ndarray
) -> np.ndarray:
    """
    Evaluate the concavity of the model (GP) posterior mean
    at the given x-values.

    Inputs:
        model: SingleTaskGP regression model trained on 1d data
        x_values: 1d numpy array specifying the inputs at which
                    to evaluate concavity of model posterior mean
    Outputs:
        is_concave_up: 1d numpy boolean array specifying which x-values
                        are in regions of model upwards concavity
    """

    x_values = torch.from_numpy(x_values)
    d2y_dx2 = posterior_mean_second_derivative(model, x_values)
    is_concave_up = d2y_dx2 > 0
    is_concave_up = is_concave_up.detach().numpy()

    return is_concave_up
