import sys
import os
from ml_tto.gpsr.lcls_tools import (
    get_lcls_tools_data,
    process_automatic_emittance_measurement_data,
)

from lcls_tools.common.image.fit import ImageProjectionFit
import os
from pydantic import ValidationError

from lcls_tools.common.data.saver import H5Saver
from ml_tto.gpsr.utils import image_snr
from gpsr.data_processing import process_images
import torch
from gpsr.datasets import ObservableDataset
from ml_tto.gpsr.utils import image_snr
import numpy as np
import matplotlib.pyplot as plt


def extract_nearest_to_evenly_spaced_x(x, y, num_points_each_side):
    """
    Extracts the sample nearest to evenly spaced target x locations around the minimum of x,y

    Args:
        x (np.ndarray): x data.
        y (np.ndarray): y data.
        num_points_each_side (int): Number of points to select on each side.

    Returns:
        (np.ndarray, np.ndarray): x_selected, y_selected
    """

    x_min = x[np.nanargmin(y)]

    # Determine x boundaries
    x_min_val = np.min(x)
    x_max_val = np.max(x)

    # Generate evenly spaced target x values on each side
    left_targets = np.linspace(
        x_min_val, x_min, num_points_each_side + 1, endpoint=False
    )[1:]
    right_targets = np.linspace(
        x_min, x_max_val, num_points_each_side + 1, endpoint=False
    )[1:]

    targets = np.concatenate([left_targets, [x_min], right_targets])

    # Select nearest actual samples to each target
    x_selected = []
    y_selected = []
    used_indices = set()

    for t in targets:
        # only select x's which have non nan y's
        x_no_nans = x[~np.isnan(y)]
        idx = np.abs(x_no_nans - t).argmin()

        # Ensure unique selections
        if idx not in used_indices:
            x_selected.append(x[idx])
            y_selected.append(y[idx])
            used_indices.add(idx)

    # if there are not enough points return all
    if len(x_selected) < num_points_each_side:
        return x[~np.isnan(y)], y[~np.isnan(y)], np.arange(len(x))[~np.isnan(y)]

    # Sort by x for clarity
    sorted_idx = np.argsort(x_selected)
    x_selected = np.array(x_selected)[sorted_idx]
    y_selected = np.array(y_selected)[sorted_idx]

    indicies = np.array([np.where(x == ele)[0] for ele in x_selected])

    return x_selected, y_selected, indicies


class CustomObservableDataset(ObservableDataset):
    def __init__(self, parameters, observables, metadata):
        super().__init__(parameters, observables)
        self.metadata = metadata


# specify pvs that we need to add to the model for GPSR
model_pvs = [
    f"QUAD:DIAG0:{val}:BCTRL"
    for val in [190, 210, 230, 270, 285, 300, 360, 370, 390, 455, 470]
] + ["TCAV:DIAG0:11:AREQ"]


def process_data_2(
    fname,
    dump_location,
    images_per_scan=5,
    min_signal_to_noise_ratio=4,
    pool_size=3,
    visualize=False,
):
    saver = H5Saver()
    all_data = saver.load(fname)
    observations = []
    parameters = []

    pool_sizes = {
        "otrdg02_pool_size": 2,
        "otrdg04_pool_size": 4,
    }

    fitter = ImageProjectionFit(validate_fit=False)

    for screen in ["OTRDG02", "OTRDG04"]:
        image_data = []
        pv_data = []
        for tcav_state in ["off", "on"]:
            data = all_data[f"{screen}_{tcav_state}"]
            beam_energy = float(
                data["environment_variables"]["BEND:DIAG0:510:BCTRL"]
            )  # in GeV/c

            # get formatted data
            formatted_data = get_lcls_tools_data(data)
            quad_values = formatted_data["quad_pv_values"]

            # sort images and pv_values, smooth images for fitting
            sorted_idx = np.argsort(quad_values)
            quad_values = quad_values[sorted_idx]
            images = formatted_data["raw_images"][sorted_idx]

            # subtract background from images
            background = data["metadata"]["beamsize_measurement"]["image_processor"][
                "background_image"
            ]
            images = np.clip(images - background.T, 0, None)

            # if OTRDG02, flip LR
            if screen == "OTRDG02":
                images = np.flip(images, (-1, -2))

            # fit smoothed images
            smoothed_images = process_images(
                images, 1, median_filter_size=3, center=True, crop=True
            )["images"]
            rms_sizes = []
            for ele in smoothed_images:
                try:
                    rms_sizes.append(fitter.fit_image(ele).rms_size)
                except (ValueError, ValidationError):
                    rms_sizes.append(np.ones(2) * np.nan)
            rms_sizes = np.array(rms_sizes).T

            if visualize:
                fig, ax = plt.subplots()
                for i in range(2):
                    ax.plot(quad_values, rms_sizes[i], ".")

            # select a subset of images
            _, _, indicies_x = extract_nearest_to_evenly_spaced_x(
                quad_values, rms_sizes[0], 5
            )
            _, _, indicies_y = extract_nearest_to_evenly_spaced_x(
                quad_values, rms_sizes[1], 5
            )
            selected_quad_pvs = np.sort(
                np.unique(np.concatenate((indicies_x, indicies_y)))
            )

            # drop the first and last as a heuristic
            selected_quad_pvs = selected_quad_pvs[1:-1]

            # select a subset of images
            B = len(selected_quad_pvs)
            if images_per_scan < B:
                idx = np.linspace(0, B - 1, images_per_scan).round().astype(int)
                selected_quad_pvs = selected_quad_pvs[idx]

            images = images[selected_quad_pvs]
            quad_values = quad_values[selected_quad_pvs]

            if visualize:
                fig2, ax2 = plt.subplots(1, images_per_scan, sharex=True, sharey=True)
                for i in range(images_per_scan):
                    ax2[i].imshow(smoothed_images[selected_quad_pvs[i]])
                for ele in quad_values:
                    ax.axvline(ele)

            image_data.append(images)

            pv_values = []
            for q_val in quad_values:
                t = [
                    data["metadata"]["image_data"][str(q_val)]["metadata"][ele]
                    for ele in model_pvs
                ]
                t += [0 if screen == "OTRDG02" else 1]
                pv_values.append(t)

            pv_values = np.array(pv_values)
            pv_data.append(pv_values)

        # need to have the same number of images in each scan to stack
        image_data = np.concat(image_data)
        pv_data = np.concat(pv_data)

        processed_images = process_images(
            image_data,
            pixel_size=1.0,
            n_stds=7,
            median_filter_size=3,
            threshold_multiplier=1.0,
            pool_size=pool_sizes[f"{screen.lower()}_pool_size"],
            crop=True,
            center=True,
        )["images"]

        # transpose images
        processed_images = np.transpose(processed_images, axes=(0, 2, 1))

        observations.append(processed_images)
        parameters.append(pv_data)

    parameters = torch.from_numpy(np.stack(parameters)).transpose(0, 1)
    observations = tuple(torch.from_numpy(ele) for ele in observations)

    # scale TCAV data based on calibration
    parameters[..., -2] *= -287759.0 / 0.5

    print("final param shape", parameters.shape)
    print("final images shape", [ele.shape for ele in observations])

    metadata = {
        "beam_energy_GeV": beam_energy,
    }
    metadata.update(pool_sizes)

    dataset = CustomObservableDataset(
        parameters,
        observations,
        metadata,
    )

    torch.save(
        dataset,
        os.path.join(dump_location, os.path.split(fname)[-1].replace(".h5", ".dset")),
    )

    return dataset
