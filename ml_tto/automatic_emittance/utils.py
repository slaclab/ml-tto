import os
import numpy as np
from lcls_tools.common.image.roi import ROI, CircularROI
from lcls_tools.common.measurements.screen_profile import (
    ScreenBeamProfileMeasurementResult,
)
from lcls_tools.common.data.model_general_calcs import bdes_to_kmod
from lcls_tools.common.data.emittance import compute_emit_bmag

from ml_tto.saver import H5Saver



def validate_beamsize_measurement_result(
    screen_measurement_result: ScreenBeamProfileMeasurementResult,
    roi: ROI,
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
            -1 * rms_size * n_stds + centroid,
            rms_size * n_stds + centroid,
            np.array((-1, 1)) * rms_size * n_stds + centroid,
            np.array((1, -1)) * rms_size * n_stds + centroid,
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
        center = np.ones(2) * roi_radius

        # calculate the max distance from the center of the ROI to the corner of the bounding box
        max_distance = np.max(
            np.array(
                [np.linalg.norm(center - corner) for corner in bounding_box_coordinates]
            )
        )

        return max_distance - roi_radius

    elif isinstance(roi, ROI):
        extent = np.array(roi.extent)
        center = extent / 2

        # calculate the max bbox extent past the edges of the ROI extent

        return np.max(
            np.abs(
                bounding_box_coordinates - center
            ) - extent / 2
        )

    else:
        raise ValueError(f"ROI type {type(roi)} is not supported for ")


def emittance_from_h5(h5_filename: str, thin_lens=False, maxiter=None): # TODO: check if any of these could be none or key not found
    """
    Parse HDF5 values needed for emittance calculation and calculate the emittance.

    Parameters
    ----------
    h5_filename : str
        The path to the HDF5 file.
    thin_lens : bool, optional
        Whether to use the thin lens approximation. Default is False.
    maxiter : int, optional
        Maximum number of iterations for the fitting optimization. Default is None.

    Returns
    -------
    dict
        The results of the emittance calculation.

    Raises
    ------
    OSError
        If the file is not found.
    """
    # Parses h5 values needed
    if os.path.exists(h5_filename):
        saver = H5Saver()
        result_dict = saver.load_from_h5(filepath=h5_filename)
    else:
        raise OSError(f"File {h5_filename} is not found.")

    inputs = {
        "quad_vals": result_dict["quadrupole_pv_values"], # dict, in kG
        "beamsizes": result_dict["rms_beamsizes"], # dict, in m
        "q_len": result_dict["metadata"]["magnet"]["metadata"]["l_eff"], # in m
        "rmat": result_dict["metadata"]["rmat"], # 2D array
        "energy": result_dict["metadata"]["energy"], # in eV
    }

    if "design_twiss" in result_dict["metadata"] and result_dict["metadata"]["design_twiss"] is not None:
        twiss_design = np.array([
            [result_dict["metadata"]["design_twiss"]["beta_x"], result_dict["metadata"]["design_twiss"]["alpha_x"]],
            [result_dict["metadata"]["design_twiss"]["beta_y"], result_dict["metadata"]["design_twiss"]["alpha_y"]]
        ])
    else:
        twiss_design = None
    inputs["twiss_design"] = twiss_design
    inputs["thin_lens"] = thin_lens
    inputs["maxiter"] = maxiter

    # Call wrapper that takes quads in machine units and beamsize in meters
    results  = compute_emit_bmag_machine_units(**inputs)
    results["metadata"] = result_dict["metadata"]

    return results

def compute_emit_bmag_machine_units(quad_vals: dict, beamsizes: dict, q_len: float, rmat: np.ndarray, energy: float, twiss_design: np.ndarray, thin_lens: bool = False, maxiter: int = None):
    """
    Wrapper for compute_emit_bmag that takes quads in machine units and beamsize in meters.

    Parameters
    ----------
    quad_vals : dict
        A dict containing the quadrupole values in kG for x and y respectively.
    beamsizes : dict
        A dict containing the beam sizes in meters for x and y respectively.
    q_len : float
        The effective length of the quadrupole in meters.
    rmat : np.ndarray
        The R-matrix. Shape (2, 2).
    energy : float
        The energy of the beam in eV.
    twiss_design : np.ndarray or None
        The design Twiss parameters. Shape (2, 2).
    thin_lens : bool, optional
        Whether to use the thin lens approximation. Default is False.
    maxiter : int, optional
        Maximum number of iterations for the optimization. Default is None.

    Returns
    -------
    dict
        The results of the emittance calculation.
    """    # Preprocessing data
    kmod_list, beamsizes_squared_list = preprocess_inputs(quad_vals, beamsizes, energy, q_len)

    # Prepare outputs
    results = {
        "emittance": [],
        "twiss_at_screen": [],
        "beam_matrix": [],
        "bmag": [] if twiss_design is not None else None,
        "quadrupole_focusing_strengths": [],
        "quadrupole_pv_values": [],
        "rms_beamsizes": [],
    }

    # Then call compute_emit_bmag
    # fit scans independently for x/y
    # only keep data that has non-nan beam sizes -- independent for x/y
    for i in range(2):
        result = compute_emit_bmag(
            k=kmod_list[i],
            beamsize_squared=beamsizes_squared_list[i],
            q_len=q_len,
            rmat=rmat[i],
            twiss_design=twiss_design[i] if twiss_design is not None else None,
            thin_lens=thin_lens,
            maxiter=maxiter,
        )

        result.update({"quadrupole_focusing_strengths": kmod_list[i]})
        result.update(
            {"quadrupole_pv_values": quad_vals[f"{i}"][~np.isnan(beamsizes[f"{i}"])]}
        )

        # add results to dict object
        for name, value in result.items():
            if name == "bmag" and value is None:
                continue
            else:
                results[name].append(value)

        results["rms_beamsizes"].append(beamsizes[f"{i}"][~np.isnan(beamsizes[f"{i}"])])

    return results

def preprocess_inputs(quad_vals: dict, beamsizes: dict, energy: float, q_len: float):
    """
    Preprocesses the inputs for compute_emit_bmag.

    Parameters
    ----------
    quad_vals : dict
        A dict containing the quadrupole values in kG for x and y respectively.
    beamsizes : dict
        A dict containing the beam sizes in meters for x and y respectively.
    energy : float
        The energy of the beam in eV.
    q_len : float
        The effective length of the quadrupole in meters.

    Returns
    -------
    tuple
        A tuple containing the list of kmod values and the list of beam sizes squared.
    """
    kmod_list = []
    beamsizes_squared_list = []

    for i in range(2):
        # Get rid of nans
        idx = ~np.isnan(beamsizes[f"{i}"])
        q = quad_vals[f"{i}"][idx]
        b = beamsizes[f"{i}"][idx]

        # Beamsizes to mm squared
        beamsizes_squared_list.append((b * 1e3) ** 2)

        # Quad values to kmod
        kmod = bdes_to_kmod(energy, q_len, q)

        # Negate for y
        if i == 1:
            kmod = -1 * kmod

        kmod_list.append(kmod)

    return kmod_list, beamsizes_squared_list

