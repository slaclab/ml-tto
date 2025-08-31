import h5py
import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from skimage.filters import threshold_triangle

def image_snr(image: np.ndarray, threshold: float = None) -> float:
    """
    Compute the signal-to-noise ratio (SNR) of a 2D image.
    
    Parameters
    ----------
    image : np.ndarray
        2D array representing the beam distribution.
    threshold : float, optional
        Pixel threshold to separate signal from noise.
        If None, use the triangle threshold method.
    
    Returns
    -------
    snr : float
        Signal-to-noise ratio (mean signal / std noise).
    """
    img = np.asarray(image)
    
    if img.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    
    pixels = img.ravel()
    
    # Automatic threshold using triangle method if not provided
    if threshold is None:
        threshold = threshold_triangle(pixels)
    
    # Define signal and noise masks
    signal_pixels = pixels[pixels > threshold]
    noise_pixels  = pixels[pixels <= threshold]
    
    if len(signal_pixels) == 0 or len(noise_pixels) == 0:
        raise ValueError("Thresholding failed: no signal or no noise pixels detected.")
    
    # Compute mean signal and std noise
    mean_signal = signal_pixels.mean()
    std_noise   = noise_pixels.std()
    
    return mean_signal / std_noise if std_noise > 0 else np.inf

def extract_nearest_to_evenly_spaced_x(x, y, num_points_each_side):
    """
    Given a 1D array of x and y values, return a sorted list of x,y pairs that
    include the minimum point for y and are evenly spaced around the minimum of y.
    Used to subsample points from an autonomous quadrupole scan.

    Args:
        x (np.ndarray): x data.
        y (np.ndarray): y data.
        num_points_each_side (int): Number of points to select on each side.

    Returns:
        (np.ndarray, np.ndarray): x_selected, y_selected
    """

    # sort by x
    sorted_idx = np.argsort(x)
    x = x[sorted_idx]
    y = y[sorted_idx]

    x_min = x[np.argmin(y)]

    # Determine x boundaries
    x_min_val = np.min(x)
    x_max_val = np.max(x)

    # Generate evenly spaced target x values on each side
    left_targets = np.linspace(x_min_val, x_min, num_points_each_side, endpoint=False)
    right_targets = np.linspace(
        x_min, x_max_val, num_points_each_side + 1, endpoint=True
    )[1:]

    targets = np.concatenate([left_targets, [x_min], right_targets])

    # Select nearest actual samples to each target
    x_selected = []
    y_selected = []
    used_indices = set()

    for t in targets:
        idx = np.abs(x - t).argmin()

        # Ensure unique selections
        if idx not in used_indices:
            x_selected.append(x[idx])
            y_selected.append(y[idx])
            used_indices.add(idx)

    # Sort by x for clarity
    sorted_idx = np.argsort(x_selected)
    x_selected = np.array(x_selected)[sorted_idx]
    y_selected = np.array(y_selected)[sorted_idx]

    return x_selected, y_selected


def select_quadrupole_scan_data(k_x, k_y, sigma_x, sigma_y, num_points_each_side):
    """
    Selects data points from a quadrupole scan by
    extracting the minimum point and evenly spaced
    points around it.

    Args:
        k (np.ndarray): Quadrupole strength data.
        sigma_x (np.ndarray): RMS x data in arbitrary units.
        sigma_y (np.ndarray): RMS y data in arbitrary units.
        num_points_each_side (int): Number of points to select on each side of a minimum.

    Returns:
        (np.ndarray, np.ndarray): k_selected, idx_selected
    """

    # extract points nearest to evenly spaced targets around the minimum of RMS x
    k_x_selected, _ = extract_nearest_to_evenly_spaced_x(
        k_x, sigma_x, num_points_each_side
    )

    # extract points nearest to evenly spaced targets around the minimum of RMS y
    k_y_selected, _ = extract_nearest_to_evenly_spaced_x(
        k_y, sigma_y, num_points_each_side
    )

    # combine sets of k_x_selected and k_y_selected into a sorted unique list
    k_selected = np.concatenate([k_x_selected, k_y_selected])
    k_selected = np.unique(k_selected)
    k_selected = np.sort(k_selected)

    return k_selected


def get_matching_indices(k, k_selected):
    """
    Get the indices of the selected k values in the original k array.

    Args:
        k (np.ndarray): Original k array.
        k_selected (np.ndarray): Selected k values.

    Returns:
        np.ndarray: Indices of the selected k values in the original k array.
    """
    return np.isin(k, k_selected).nonzero()[0]