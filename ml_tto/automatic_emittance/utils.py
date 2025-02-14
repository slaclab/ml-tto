import numpy as np
from lcls_tools.common.image.roi import ROI, CircularROI
from lcls_tools.common.measurements.screen_profile import (
    ScreenBeamProfileMeasurementResult,
)


def validate_beamsize_measurement_result(
    screen_measurement_result: ScreenBeamProfileMeasurementResult,
    min_log10_intensity: float = 3.0,
    n_stds: float = 2.0,
):
    """
    Validate the beamsize measurement result. Set rms/centroid values to NaN if the total intensity
    is below a threshold or if the bounding box penalty is positive.

    Parameters
    ----------
    screen_measurement_result : ScreenBeamProfileMeasurementResult
        The beamsize measurement result.
    min_log10_intensity : float, optional
        Minimum log10 intensity threshold, default is 3.0.
    n_stds : float, optional
        Number of standard deviations to use for the bounding box, default is 2.0.

    Returns
    -------
    tuple
        The validated beamsize measurement result, bounding box penalties, and the log10 total intensity.

    """
    roi = screen_measurement_result.metadata.image_processor.roi

    # get the total intensity of the images
    log10_total_intensity = np.log10(screen_measurement_result.total_intensities)

    # convert rms_sizes and centroids to numpy float32 arrays
    screen_measurement_result.rms_sizes = screen_measurement_result.rms_sizes.astype(
        dtype=np.float32
    )
    screen_measurement_result.centroids = screen_measurement_result.centroids.astype(
        dtype=np.float32
    )

    # calculate bounding box penalty for each image
    bb_penalties = []
    rms_sizes = screen_measurement_result.rms_sizes
    centroids = screen_measurement_result.centroids
    for i in range(len(screen_measurement_result.rms_sizes)):
        bb_penalties.append(
            calculate_bounding_box_penalty(
                roi,
                calculate_bounding_box_coordinates(rms_sizes[i], centroids[i], n_stds),
            )
        )

    # set rms/centroid values to NaN if the total intensity is below the
    # threshold or if the bounding box penalty is positive
    # also set the bb_penalties to Nan if the total intensity is below the threshold
    for i in range(len(screen_measurement_result.rms_sizes)):
        if log10_total_intensity[i] < min_log10_intensity:
            bb_penalties[i] = np.empty_like(bb_penalties[i], dtype=np.float32) * np.nan

        if log10_total_intensity[i] < min_log10_intensity or bb_penalties[i] > 0:
            screen_measurement_result.rms_sizes[i] = (
                np.empty_like(rms_sizes[i], dtype=np.float32) * np.nan
            )
            screen_measurement_result.centroids[i] = (
                np.empty_like(centroids[i], dtype=np.float32) * np.nan
            )

    return screen_measurement_result, bb_penalties, log10_total_intensity


def calculate_bounding_box_coordinates(
    rms_size: np.ndarray, centroid: np.ndarray, n_stds: float
) -> np.ndarray:
    """
    Calculate the corners of a bounding box given the fit results.

    Parameters
    ----------
    rms_size : np.ndarray, shape (2,)
        The root mean square size of the beam.
    centroid : np.ndarray, shape (2,)
        The centroid of the beam.
    n_stds : float
        Number of standard deviations to use for the bounding box.

    Returns
    -------
    np.ndarray, shape (4, 2)
        The calculated bounding box coordinates.
    """
    return np.array(
        [
            -1 * rms_size * n_stds / 2 + centroid,
            rms_size * n_stds / 2 + centroid,
            np.array((-1, 1)) * rms_size * n_stds / 2 + centroid,
            np.array((1, -1)) * rms_size * n_stds / 2 + centroid,
        ]
    )


def calculate_bounding_box_penalty(
    roi: ROI, bounding_box_coordinates: np.ndarray
) -> float:
    """
    Calculate the penalty based on the maximum distance between the center of the ROI
    and the beam bounding box corners.

    Parameters
    ----------
    roi : ROI
        Region of interest, can be either CircularROI or ROI.
    bounding_box_coordinates : np.ndarray, shape (4, 2)
        The bounding box coordinates.

    Returns
    -------
    float
        The calculated penalty value.

    Raises
    ------
    ValueError
        If the ROI type is not supported.
    """
    if isinstance(roi, CircularROI):
        roi_radius = roi.radius[0]
    elif isinstance(roi, ROI):
        roi_radius = np.min(np.array(roi.extent) / 2)
    else:
        raise ValueError(f"ROI type {type(roi)} is not supported for ")

    roi_center = roi.center

    # calculate the max distance from the center of the ROI to the corner of the bounding box
    max_distance = np.max(
        np.array(
            [np.linalg.norm(roi_center - corner) for corner in bounding_box_coordinates]
        )
    )

    return max_distance - roi_radius
