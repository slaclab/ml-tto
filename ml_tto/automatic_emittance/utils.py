import os
from collections.abc import Mapping

import numpy as np
from ml_tto.roi import ROI, CircularROI
from slac_measurements.screen_profile import (
    ScreenBeamProfileMeasurementResult,
)
from slac_model.model_general_calcs import bdes_to_kmod, build_quad_rmat
from slac_measurements.emittance import compute_emit_bmag, normalize_emittance
from slac_measurements.emittance_measurement import (
    QuadScanEmittanceResult,
)
from ml_tto.io import H5Saver


def quad_scan_result_from_dict(data: dict) -> QuadScanEmittanceResult:
    """Upgrade historical geometric-emittance results to the SLAC result schema."""
    data = dict(data)
    data.pop("environment_variables", None)
    if "twiss_at_screen" in data:
        data["twiss"] = data.pop("twiss_at_screen")
    for key in (
        "quadrupole_focusing_strengths",
        "quadrupole_pv_values",
        "bmag",
        "phase_advances",
        "twiss",
        "rms_beamsizes",
    ):
        if isinstance(data.get(key), Mapping):
            values = data[key]
            data[key] = [values[key] for key in sorted(values, key=int)]
    if "geometric_emittance" not in data:
        if "energy" not in data:
            data["energy"] = data["metadata"]["energy"]
        data["geometric_emittance"] = np.asarray(data["emittance"])
        data["emittance"] = normalize_emittance(
            data["geometric_emittance"], data["energy"] * 1e-9
        )
        data["rms_beamsizes"] = [
            np.asarray(size) * 1e6 for size in data["rms_beamsizes"]
        ]
    if "twiss_at_reconstruction" not in data:
        matrix = np.asarray(data["beam_matrix"])
        twiss = matrix * np.array([1.0, -1.0, 1.0])
        data["twiss_at_reconstruction"] = np.expand_dims(
            twiss / np.asarray(data["geometric_emittance"]).reshape(2, 1), -2
        )
    return QuadScanEmittanceResult(**data)


def _to_1d_float_array(value):
    """Convert supported input values to a 1D float array.

    Parameters
    ----------
    value : Any
        Input value that can be a scalar, sequence, mapping, numpy array,
        or None.

    Returns
    -------
    numpy.ndarray
        One-dimensional float array representation of the input.
    """
    if value is None:
        return np.array([], dtype=float)

    # Handle pandas-like vectors loaded from QuadScanEmittanceResult metadata.
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy(), dtype=float).reshape(-1)

    if isinstance(value, np.ndarray):
        return value.astype(float).reshape(-1)

    if isinstance(value, Mapping):
        keys = list(value.keys())

        def _sort_key(k):
            k_str = str(k)
            if k_str.lstrip("-").isdigit():
                return (0, int(k_str))
            return (1, k_str)

        sorted_keys = sorted(keys, key=_sort_key)
        values = [value[k] for k in sorted_keys]
        return np.array(values, dtype=float).reshape(-1)

    if isinstance(value, (list, tuple)):
        return np.array(value, dtype=float).reshape(-1)

    return np.array([value], dtype=float)


def _get_result_attr(result, key, default=None):
    """Read an attribute from an object or key from a mapping.

    Parameters
    ----------
    result : Any
        Source object or mapping.
    key : str
        Attribute/key name to retrieve.
    default : Any, optional
        Fallback value if key/attribute does not exist.

    Returns
    -------
    Any
        Retrieved value or default.
    """

    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _extract_raw_scan_arrays(emittance_result):
    """Extract raw scan arrays used for emittance scan plotting.

    Parameters
    ----------
    emittance_result : QuadScanEmittanceResult or dict
        Emittance result with metadata containing raw Xopt scan values.

    Returns
    -------
    tuple
        (metadata, scan_values, x_rms_micron_sq, y_rms_micron_sq), where arrays
        are length-matched to the shortest available sequence.

    Raises
    ------
    ValueError
        If required raw scan arrays are missing.
    """

    metadata = _get_result_attr(emittance_result, "metadata", {})
    x_data = metadata.get("X", {}).get("data", {})

    scan_values = _to_1d_float_array(x_data.get("k", metadata.get("scan_values", [])))
    x_rms_micron_sq = _to_1d_float_array(x_data.get("x_rms_micron_sq", []))
    y_rms_micron_sq = _to_1d_float_array(x_data.get("y_rms_micron_sq", []))

    if len(scan_values) == 0 or len(x_rms_micron_sq) == 0 or len(y_rms_micron_sq) == 0:
        raise ValueError(
            "Could not find raw scan data in result metadata. "
            "Expected metadata['X']['data'] to include 'k', 'x_rms_micron_sq', and 'y_rms_micron_sq'."
        )

    npts = min(len(scan_values), len(x_rms_micron_sq), len(y_rms_micron_sq))
    return metadata, scan_values[:npts], x_rms_micron_sq[:npts], y_rms_micron_sq[:npts]


def _compute_cutoff_um(metadata, rms_micron_sq):
    """Compute the beam-size cutoff in microns from scan metadata.

    Parameters
    ----------
    metadata : dict
        Emittance metadata dictionary.
    rms_micron_sq : numpy.ndarray
        Beam-size-squared values in micron^2.

    Returns
    -------
    float or None
        Cutoff in microns, or None if cutoff metadata is unavailable.
    """

    beamsize_cutoff_max = metadata.get("beamsize_cutoff_max", None)
    min_beamsize_cutoff = metadata.get("min_beamsize_cutoff", None)

    if beamsize_cutoff_max is None or min_beamsize_cutoff is None:
        return None

    finite_mask = np.isfinite(rms_micron_sq) & (rms_micron_sq > 0.0)
    if not np.any(finite_mask):
        return None

    min_size_micron_sq = np.nanmin(rms_micron_sq[finite_mask])
    cutoff_micron = max(
        float(beamsize_cutoff_max) * np.sqrt(min_size_micron_sq),
        float(min_beamsize_cutoff),
    )
    return cutoff_micron


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

        return np.max(np.abs(bounding_box_coordinates - center) - extent / 2)

    else:
        raise ValueError(f"ROI type {type(roi)} is not supported for ")


def emittance_from_h5(h5_filename: str, thin_lens=False, maxiter=None):
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
        result_dict = saver.load(filepath=h5_filename)
    else:
        raise OSError(f"File {h5_filename} is not found.")

    def ordered_arrays(value):
        if isinstance(value, Mapping):
            value = [value[key] for key in sorted(value, key=int)]
        return [np.asarray(item) for item in value]

    beam_sizes = ordered_arrays(result_dict["rms_beamsizes"])
    if "geometric_emittance" in result_dict:
        beam_sizes = [size * 1e-6 for size in beam_sizes]
    inputs = {
        "quad_vals": ordered_arrays(result_dict["quadrupole_pv_values"]),  # in kG
        "beamsizes": beam_sizes,  # in m
        "q_len": result_dict["metadata"]["magnet"]["metadata"]["l_eff"],  # in m
        "rmat": result_dict["metadata"]["rmat"],  # 2D array
        "energy": result_dict["metadata"]["energy"],  # in eV
    }

    if (
        "design_twiss" in result_dict["metadata"]
        and result_dict["metadata"]["design_twiss"] is not None
    ):
        twiss_design = np.array(
            [
                [
                    result_dict["metadata"]["design_twiss"]["beta_x"],
                    result_dict["metadata"]["design_twiss"]["alpha_x"],
                ],
                [
                    result_dict["metadata"]["design_twiss"]["beta_y"],
                    result_dict["metadata"]["design_twiss"]["alpha_y"],
                ],
            ]
        )
    else:
        twiss_design = None
    inputs["twiss_design"] = twiss_design
    inputs["thin_lens"] = thin_lens
    inputs["maxiter"] = maxiter

    # Call wrapper that takes quads in machine units and beamsize in meters
    results = compute_emit_bmag_machine_units(**inputs)
    results["metadata"] = result_dict["metadata"]
    results["energy"] = inputs["energy"]

    return QuadScanEmittanceResult(**results)


def compute_emit_bmag_machine_units(
    quad_vals: list,
    beamsizes: list,
    q_len: float,
    rmat: np.ndarray,
    energy: float,
    twiss_design: np.ndarray,
    thin_lens: bool = False,
    maxiter: int = None,
    rmat_includes_quad: bool = False,
):
    """
    Wrapper for compute_emit_bmag that takes quads in machine units and beamsize in meters.

    Parameters
    ----------
    quad_vals : list
        A list of two arrays containing the quadrupole values in kG for x and y respectively.
    beamsizes : list
        A list of two arrays containing the beam sizes in meters for x and y respectively.
    q_len : float
        The effective length of the quadrupole in meters.
    rmat : np.ndarray
        The R-matrix. Shape (2, 2, 2).
    energy : float
        The energy of the beam in eV.
    twiss_design : np.ndarray or None
        The design Twiss parameters. Shape (2, 2).
    thin_lens : bool, optional
        Whether to use the thin lens approximation. Default is False.
    maxiter : int, optional
        Maximum number of iterations for the optimization. Default is None.
    rmat_includes_quad : bool, optional
        If True, rmat has shape (2, n_steps, 2, 2) and already includes the quad.

    Returns
    -------
    dict
        The results of the emittance calculation.
    """  # Preprocessing data
    kmod_list, beamsizes_squared_list = preprocess_inputs(
        quad_vals, beamsizes, energy, q_len
    )

    # Prepare outputs
    results = {
        "emittance": [],
        "geometric_emittance": [],
        "twiss": [],
        "twiss_at_reconstruction": [],
        "beam_matrix": [],
        "bmag": [] if twiss_design is not None else None,
        "phase_advances": [] if twiss_design is not None else None,
        "quadrupole_focusing_strengths": [],
        "quadrupole_pv_values": [],
        "rms_beamsizes": [],
    }

    # Then call compute_emit_bmag
    # fit scans independently for x/y
    # only keep data that has non-nan beam sizes -- independent for x/y
    for i in range(2):
        if rmat_includes_quad:
            total_rmat = rmat[i][~np.isnan(beamsizes[i])]
        else:
            quad_rmat = build_quad_rmat(kmod_list[i], q_len, thin_lens=thin_lens)
            total_rmat = rmat[i] @ quad_rmat
        result = compute_emit_bmag(
            beamsize_squared=np.expand_dims(beamsizes_squared_list[i], -1),
            rmat=total_rmat,
            twiss_design=(
                np.expand_dims(twiss_design[i], -2)
                if twiss_design is not None
                else None
            ),
            maxiter=maxiter,
        )
        result["geometric_emittance"] = result["emittance"]
        result["emittance"] = normalize_emittance(result["emittance"], energy * 1e-9)

        result.update({"quadrupole_focusing_strengths": kmod_list[i]})
        result.update({"quadrupole_pv_values": quad_vals[i][~np.isnan(beamsizes[i])]})

        # add results to dict object
        for name, value in result.items():
            if name in {"bmag", "phase_advances"} and value is None:
                continue
            else:  # beam matrix and emittance get appended
                results[name].append(value)

        results["rms_beamsizes"].append(beamsizes[i][~np.isnan(beamsizes[i])] * 1e6)

    return results


def preprocess_inputs(quad_vals: list, beamsizes: list, energy: float, q_len: float):
    """
    Preprocesses the inputs for compute_emit_bmag.

    Parameters
    ----------
    quad_vals : list
        A list of two arrays containing the quadrupole values in kG for x and y respectively.
    beamsizes : dict
        A list of two arrays containing the beam sizes in meters for x and y respectively.
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
        idx = ~np.isnan(beamsizes[i])
        q = quad_vals[i][idx]
        b = beamsizes[i][idx]

        # Beamsizes to mm squared
        beamsizes_squared_list.append((b * 1e3) ** 2)

        # Quad values to kmod
        kmod = bdes_to_kmod(energy, q_len, q)

        # Negate for y
        if i == 1:
            kmod = -1 * kmod

        kmod_list.append(kmod)

    return kmod_list, beamsizes_squared_list
