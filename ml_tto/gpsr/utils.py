from typing import Tuple
import torch
import numpy as np

from skimage.filters import threshold_triangle
from lcls_tools.common.data.model_general_calcs import bmag
import lightning as L
from gpsr.modeling import GPSRLattice
from cheetah.accelerator import Segment, CustomTransferMap


def get_beam_stats(reconstructed_beam, gpsr_model, design_twiss=None):
    """
    Compute second order moment related statistics of the reconstructed beam distribution

    Parameters
    ----------
    reconstructed_beam: Beam
        The reconstructed beam object.
    gpsr_model: GPSRModel
        The GPSR model used for the reconstruction.
    design_twiss: list, optional
        The design Twiss parameters. Should have the form [beta_x, alpha_x, beta_y, alpha_y].

    Returns
    -------
    dict
        A dictionary containing the computed beam statistics. The dictionary has the following elements
        - norm_emit_x: Normalized emittance along x in m.rad
        - norm_emit_y: Normalized emittance along y in m.rad
        - beta_x: Beta function along x in m
        - beta_y: Beta function along y in m
        - alpha_x: Alpha function along x
        - alpha_y: Alpha function along y
        - screen_distribution: The distribution of the beam at the screen for each quadrupole focusing strength
        - twiss_at_screen: The Twiss parameters at the screen for each quadrupole focusing strength
        - rms_sizes: The RMS sizes of the beam at the screen for each quadrupole focusing strength
        - sigma_matrix: The covariance matrix of the reconstructed beam
        - bmag: The bmag matching parameter for each quadrupole focusing strength

    """

    # track the reconstructed beam to the screen
    final_beam = gpsr_model.lattice.lattice.track(reconstructed_beam)

    # compute the twiss functions
    twiss_at_screen = [
        torch.stack(
            (
                final_beam.beta_x,
                final_beam.alpha_x,
                (1 + final_beam.alpha_x**2) / final_beam.beta_x,
            )
        )
        .T.detach()
        .numpy(),
        torch.stack(
            (
                final_beam.beta_y,
                final_beam.alpha_y,
                (1 + final_beam.alpha_y**2) / final_beam.beta_y,
            )
        )
        .T.detach()
        .numpy(),
    ]

    if design_twiss is not None:
        bmag_val = bmag(
            [
                final_beam.beta_x.detach().numpy(),
                final_beam.alpha_x.detach().numpy(),
                final_beam.beta_y.detach().numpy(),
                final_beam.alpha_y.detach().numpy(),
            ],
            design_twiss,
        )

    # compute the rms sizes
    rms_sizes = [
        final_beam.sigma_x.detach().numpy(),
        final_beam.sigma_y.detach().numpy(),
    ]

    # compute the reconstructed beam matrix
    cov = torch.cov(reconstructed_beam.particles.T)
    beam_matrix = (
        torch.stack(
            (
                torch.triu(cov[:2, :2]).flatten()[
                    torch.triu(cov[:2, :2]).flatten() != 0
                ],
                torch.triu(cov[2:4, 2:4]).flatten()[
                    torch.triu(cov[2:4, 2:4]).flatten() != 0
                ],
            )
        )
        .detach()
        .numpy()
    )

    results = {
        "emittance": np.array(
            [
                reconstructed_beam.emittance_x.detach().cpu().numpy(),
                reconstructed_beam.emittance_y.detach().cpu().numpy(),
            ]
        ).reshape(2, 1)
        * 1e6,
        "beta_x": reconstructed_beam.beta_x,
        "beta_y": reconstructed_beam.beta_y,
        "alpha_x": reconstructed_beam.alpha_x,
        "alpha_y": reconstructed_beam.alpha_y,
        "screen_distribution": final_beam,
        "twiss_at_screen": twiss_at_screen,
        "rms_beamsizes": rms_sizes,
        "beam_matrix": beam_matrix,
    }

    if design_twiss is not None:
        results["bmag"] = bmag_val
    else:
        results["bmag"] = None

    return results


class RMatLattice(GPSRLattice):
    """
    GPSR lattice that uses a transfer matrix to propagate the beam and a single screen to observe the beam.
    """

    def __init__(self, rmat, screen, fit_threshold=False):
        """
        Initializes the RMatLattice with a transfer matrix and a screen.

        Parameters
        ----------
        rmat: torch.Tensor
            The 6D transfer matrix (cheetah coordinate system) from the reconstruction location
            to the diagnostic screen. Should have the shape (B, 6, 6) where B is the batch size
            corresponding to the number of measurements
        screen: Screen
            The cheetah screen object for measurements.
        fit_threshold: bool
            Whether to fit the threshold parameter which clips the lower intensity of the beam image.

        """
        super().__init__()
        self.fit_threshold = fit_threshold

        if self.fit_threshold:
            self.register_parameter(
                "threshold", torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            )
        self.lattice = Segment(
            [
                CustomTransferMap(rmat),
                screen,
            ]
        )

    def set_lattice_parameters(self, x: torch.Tensor) -> None:
        pass

    def track_and_observe(self, beam) -> Tuple[torch.Tensor, ...]:
        """
        tracks beam through the lattice and returns observations

        Returns
        -------
        results: Tuple[Tensor]
            Tuple of results from each measurement path
        """
        self.lattice.elements[-1].pixel_size = self.lattice.elements[-1].pixel_size.to(
            beam.x
        )
        beam.particle_charges = torch.ones_like(beam.x).to(device=beam.x.device)
        self.lattice.track(beam)

        observations = self.lattice.elements[-1].reading.transpose(-1, -2)

        # clip observations
        if self.fit_threshold:
            observations = torch.clip(observations - self.threshold * 1e-3, 0, None)

        return tuple(observations.unsqueeze(0))


class CustomLeakyReLU(torch.nn.Module):
    def forward(self, x):
        return 2 * torch.nn.LeakyReLU(negative_slope=0.1)(x)


class MetricTracker(L.Callback):
    def __init__(self):
        self.training_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.training_loss.append(trainer.callback_metrics["loss"].item())

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