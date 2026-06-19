import numpy as np
from matplotlib import pyplot as plt
import torch

from lcls_tools.common.frontend.plotting.image import (
    plot_image_projection_fit,
)

from ml_tto.automatic_emittance.scan_cropping import (
    crop_scan,
    posterior_mean_concavity,
)
from ml_tto.automatic_emittance.utils import (
    _compute_cutoff_meters,
    _extract_raw_scan_arrays,
)


def plot_screen_profile_measurement(measurement):
    """Plot a fitted screen profile measurement.

    Parameters
    ----------
    measurement : ScreenBeamProfileMeasurement
        Screen profile measurement object containing the image source,
        image processor, and fit model.

    Returns
    -------
    tuple
        Matplotlib figure and axes returned by
        lcls_tools.common.frontend.plotting.image.plot_image_projection_fit.
    """

    result = measurement.beam_fit.fit_image(
        measurement.image_processor.process(measurement.beam_profile_device.image)
    )

    return plot_image_projection_fit(result)


def plot_emittance_measurement(emittance_result):
    """Plot raw x/y beam sizes and cropping diagnostics.

    Parameters
    ----------
    emittance_result : QuadScanEmittanceResult or dict
        Emittance result object containing metadata['X']['data'] with
        raw scan values and beam sizes.

    Returns
    -------
    tuple
        Matplotlib (figure, axes) for x/y scan dimensions.
    """

    metadata, scan_values, x_rms_micron_sq, y_rms_micron_sq = _extract_raw_scan_arrays(
        emittance_result
    )

    dim_inputs = [
        ("x", x_rms_micron_sq),
        ("y", y_rms_micron_sq),
    ]
    dim_data = []

    # Process x/y uniformly to keep cropping and plotting behavior consistent.
    for dim_name, rms_micron_sq in dim_inputs:
        raw_micron = np.where(rms_micron_sq >= 0.0, np.sqrt(rms_micron_sq), np.nan)
        cutoff_m = _compute_cutoff_meters(metadata, rms_micron_sq)

        (
            _,
            beam_sizes_cropped_m,
            concavity_mask,
            cutoff_mask,
            _,
            model,
        ) = crop_scan(
            scan_values=scan_values,
            beam_sizes=raw_micron * 1e-6,
            cutoff_max=cutoff_m,
        )

        retained = ~np.isnan(beam_sizes_cropped_m)
        removed = (concavity_mask | cutoff_mask) & np.isfinite(raw_micron)

        dim_data.append(
            {
                "name": dim_name,
                "raw_micron": raw_micron,
                "retained": retained,
                "removed": removed,
                "cutoff_m": cutoff_m,
                "model": model,
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, dim in zip(axes, dim_data):
        dim_name = dim["name"]
        raw_micron = dim["raw_micron"]
        retained = dim["retained"]
        removed = dim["removed"]
        cutoff_m = dim["cutoff_m"]
        model = dim["model"]

        if model is not None:
            fit_x_t = torch.linspace(
                float(scan_values.min()), float(scan_values.max()), 100
            ).reshape(-1, 1)
            fit_y_sq_m = model.posterior(fit_x_t).mean.detach().numpy().flatten()
            fit_concavity_values = posterior_mean_concavity(model, fit_x_t.numpy().flatten())
            fit_is_concave_up = fit_concavity_values > 0
            fit_y_micron = np.sqrt(np.clip(fit_y_sq_m, a_min=0.0, a_max=None)) * 1e6
            fit_y_up = np.ma.masked_array(fit_y_micron, mask=~fit_is_concave_up)
            fit_y_down = np.ma.masked_array(fit_y_micron, mask=fit_is_concave_up)

            ax.plot(
                fit_x_t.numpy().flatten(),
                fit_y_up,
                ls="--",
                c="C1",
                label="concave up",
                zorder=1,
            )
            ax.plot(
                fit_x_t.numpy().flatten(),
                fit_y_down,
                ls="--",
                c="C2",
                label="concave down",
                zorder=1,
            )

        ax.scatter(
            scan_values,
            raw_micron,
            s=35,
            c="0.8",
            label="raw",
            zorder=1,
        )
        ax.scatter(
            scan_values[retained],
            raw_micron[retained],
            s=45,
            c="C0",
            label="retained",
            zorder=3,
        )
        ax.scatter(
            scan_values[removed],
            raw_micron[removed],
            s=50,
            marker="x",
            c="C3",
            label="cropped",
            zorder=4,
        )

        if cutoff_m is not None:
            ax.axhline(
                cutoff_m * 1e6,
                ls=":",
                c="k",
                label="cutoff",
                zorder=2,
            )

        ax.set_title(f"{dim_name.upper()} RMS")
        ax.set_xlabel("Quad value (machine units)")
        ax.set_ylabel("Beam size (micron)")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("Emittance Measurement Scan Cropping")
    fig.tight_layout()

    return fig, axes


