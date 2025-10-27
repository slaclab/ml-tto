import logging

logger = logging.getLogger("utils")

import numpy as np


def extract_nearest_to_evenly_spaced_x(x, y, num_points_each_side):
    """
    Extracts the sample nearest to evenly spaced target x locations around the minimum of x,y

    Args:
        center_index (int): Index of the center (typically np.argmin(y)).
        x (np.ndarray): x data.
        y (np.ndarray): y data.
        num_points_each_side (int): Number of points to select on each side.

    Returns:
        (np.ndarray, np.ndarray): x_selected, y_selected
    """

    logger.debug(
        f"Called extract_nearest_to_evenly_spaced_x with {len(x)} points, num_points_each_side={num_points_each_side}"
    )

    x_min = x[np.argmin(y)]
    logger.debug(f"Minimum y at x = {x_min}")

    # Determine x boundaries
    x_min_val = np.min(x)
    x_max_val = np.max(x)

    # Generate evenly spaced target x values on each side
    left_targets = np.linspace(x_min_val, x_min, num_points_each_side, endpoint=False)
    right_targets = np.linspace(
        x_min, x_max_val, num_points_each_side + 1, endpoint=True
    )[1:]

    targets = np.concatenate([left_targets, [x_min], right_targets])
    logger.debug(f"Generated {len(targets)} target points around x_min")

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

    logger.debug(f"Selected {len(x_selected)} points nearest to targets")

    return x_selected, y_selected
