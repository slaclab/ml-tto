from copy import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from lcls_tools.common.image.fit import ImageProjectionFitResult


def plot_image_projection_fit(result: ImageProjectionFitResult, n_stds: float = 4.0):
    """
    plot image and projection data for validation
    """
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(4, 9)

    image = result.image
    c = ax[0].imshow(image)
    fig.colorbar(c, ax=ax[0])

    projections = [np.sum(image, axis=0), np.sum(image, axis=1)]
    centroid = np.array([ele["mean"] for ele in result.projection_fit_parameters])
    rms_size = np.array([ele["sigma"] for ele in result.projection_fit_parameters])

    ax[0].plot(*centroid, "+r")

    # plot bounding box
    ax[0].add_patch(
        Rectangle(
            (centroid - n_stds * rms_size),
            2 * n_stds * rms_size[0],
            2 * n_stds * rms_size[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
    )

    # plot data and model fit
    for i in range(2):
        fit_params = copy(result.projection_fit_parameters[i])
        fit_params.update({"stnr": result.signal_to_noise_ratio[i]})

        text_info = "\n".join([f"{name}: {val:.2f}" for name, val in fit_params.items()])
        text_info += (
            "\n" + "extent " + ",".join([f"{ele:.2f}" for ele in result.beam_extent[i]])
        )
        ax[i + 1].text(
            0.01,
            0.99,
            text_info,
            transform=ax[i + 1].transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        x = np.arange(len(projections[i]))

        ax[i + 1].plot(projections[i], label="data")
        ax[i + 1].plot(
            result.projection_fit_method.forward(x, fit_params), label="model fit"
        )

    return fig, ax
