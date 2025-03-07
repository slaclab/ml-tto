import numpy as np
from matplotlib import pyplot as plt
from typing import Optional

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch import fit_gpytorch_mll
import torch
from gpytorch import ExactMarginalLogLikelihood


def evaluate_concavity(x: np.ndarray, y: np.ndarray):
    """
    Identifies which points in the data are in regions of upward concavity
    using posterior mean of GP fit to the data.

    NOTE: y must not contain NaNs

    Inputs:
        x: 1d numpy array containing scan inputs
        y: numpy array (same shape as x) containing corresponding scan outputs
    Outputs:
        data_concavity: truth table specifying which points in the input data
                        are in a region of upward concavity according to
                        posterior mean of GP fit to the data
        fit_x: linspace over which the GP posterior mean has been evaluated
        fit_y: GP posterior mean evaluated on the linspace fit_x
        fit_concavity: truth table specifying which points fit_x/fit_y are
                        in a region of upward concavity according to GP
                        posterior mean
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

    # evaluate the concavity of the GP fit at the training data points
    def gp_mean(quad_values):
        return model.posterior(quad_values.reshape(-1, 1)).mean.sum()

    hess_data = torch.diag(
        torch.autograd.functional.hessian(gp_mean, train_X.flatten())
    )
    data_concavity = hess_data > 0
    data_concavity = data_concavity.detach().numpy()

    # evaluate the GP fit and its concavity on a linspace
    fit_x = torch.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    fit_y = model.posterior(fit_x).mean.detach().numpy().flatten()
    hess_fit = torch.diag(
        torch.autograd.functional.hessian(gp_mean, fit_x.flatten())
    )
    fit_concavity = hess_fit > 0
    fit_concavity = fit_concavity.detach().numpy()
    fit_x = fit_x.detach().numpy().flatten()

    return data_concavity, fit_x, fit_y, fit_concavity


def crop_scans(
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
        scan_values: Cropped scan_values
        beam_sizes: Cropped beam_sizes
    """

    # remove nans and copy input data before making edits
    scan_values = scan_values[~np.isnan(beam_sizes)]
    beam_sizes = beam_sizes[~np.isnan(beam_sizes)]
    scan_values_cropped = np.copy(scan_values)
    beam_sizes_cropped = np.copy(beam_sizes)

    # identify data points where GP fit to data is concave up
    data_concavity, fit_x, fit_y, fit_concavity = evaluate_concavity(
        scan_values_cropped, beam_sizes_cropped**2
    )
    # set beam size data to nan where concavity is not upward
    concavity_mask = ~data_concavity
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
        fit_y_up = np.ma.masked_array(fit_y, mask=~fit_concavity)
        fit_y_down = np.ma.masked_array(fit_y, mask=fit_concavity)
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
            scan_values[~cutoff_mask * ~data_concavity],
            beam_sizes[~cutoff_mask * ~data_concavity] ** 2,
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
        plt.ylabel("Beam size^2 (m^2)")
        plt.xlabel("Quad value (machine units)")
        plt.title("Inflection point crop")
        plt.tight_layout()

    return scan_values_cropped, beam_sizes_cropped
