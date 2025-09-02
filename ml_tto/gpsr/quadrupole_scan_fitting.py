from typing import Tuple
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from lcls_tools.common.data.model_general_calcs import bmag

from gpsr.modeling import GPSR, GPSRLattice
from gpsr.train import LitGPSR
from gpsr.beams import NNParticleBeamGenerator, NNTransform
from gpsr.datasets import QuadScanDataset
from gpsr.data_processing import process_images
from cheetah.accelerator import Screen
from cheetah.particles import ParticleBeam
from cheetah.accelerator import CustomTransferMap, Segment

import lightning as L
import time


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


def get_beam_fraction(beam_distribution, beam_fraction, particle_slices=None):
    """
    Get the core of the beam according to 6D normalized beam coordinates.
    Particles from the beam distribution are scaled to normalized coordinates
    via the covariance matrix. Then they are sorted by distance from the origin.

    Parameters
    ----------
    beam_distribution: ParticleBeam
        Cheetah ParticleBeam object representing the beam distribution.
    beam_fraction: float
        The fraction of the beam to keep from 0 - 1.
    particle_slices: list[slice] or slice, optional
        List of slices to apply along the particle coordinates

    Returns
    -------
    ParticleBeam
        The core of the beam represented as a Cheetah ParticleBeam object.

    """
    # extract macroparticles
    macroparticles = beam_distribution.particles.detach()[..., :6]

    if isinstance(particle_slices, list):
        data = macroparticles[..., particle_slices]
    elif isinstance(particle_slices, slice):
        data = macroparticles[..., particle_slices]
    else:
        data = macroparticles

    # calculate covariance matrix
    cov = torch.cov(data.T)

    # get inverse cholesky decomp -- if it fails, add a small value to the
    # diagonal and try again
    try:
        cov_cholesky = torch.linalg.cholesky(cov)
    except torch._C._LinAlgError as e:
        cov_cholesky = torch.linalg.cholesky(cov + 1e-8 * torch.eye(6))

    # transform particles to normalized coordinates
    t_data = (torch.linalg.inv(cov_cholesky) @ data.T).T

    # sort particles by their distance from the origin and hold onto a fraction of them
    J = torch.linalg.norm(t_data, axis=1)
    sort_idx = torch.argsort(J)
    frac_coords = macroparticles[sort_idx][: int(len(data) * beam_fraction)]

    # create a beam distribution to return
    frac_coords = torch.hstack((frac_coords, torch.ones((len(frac_coords), 1))))
    frac_particle = ParticleBeam(
        frac_coords.to(beam_distribution.energy),
        energy=beam_distribution.energy,
    )

    return frac_particle


class CustomLeakyReLU(torch.nn.Module):
    def forward(self, x):
        return 2 * torch.nn.LeakyReLU(negative_slope=0.1)(x)


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


class MetricTracker(L.Callback):
    def __init__(self):
        self.training_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.training_loss.append(trainer.callback_metrics["loss"].item())


def gpsr_fit_file(
    fname: str,
    data_slice: slice = None,
    save_location: str = None,
    visualize: bool = False,
    max_pixels: int = 1e5,
    n_stds: int = 5,
    threshold_multiplier=1.2,
    **kwargs,
):
    """
    Use GPSR to fit the beam distribution from emittance measurements taken at LCLS/LCLS-II.
    Supports dump files from matlab (.mat) and lcls-tools (.h5, .hdf5).

    Parameters
    ----------
    fname: str
        The path to the data file.
    pool_size: int, optional
        Size of pooling layer to compress images to smaller sizes for speeding up GPSR fitting.
    data_slice: slice, optional
        A slice object to select a subset of the data for fitting.
    save_location: str, optional
        The location to save diagnostic plots.
    visualize: bool, optional
        Whether to visualize the diagnostic plots.
    kwargs:
        Optional arguments passed to `gpsr_fit_quad_scan`

    """

    if os.path.splitext(fname)[-1] == ".mat":
        from ml_tto.gpsr.matlab import get_matlab_data

        data = get_matlab_data(fname)

        # matlab data has multiple shots that we avg over so we multiply the max pixels by n_shots
        max_pixels *= data["images"].shape[0]

    elif os.path.splitext(fname)[-1] in [".h5", ".hdf5"]:
        from ml_tto.gpsr.lcls_tools import get_lcls_tools_data_from_file

        data = get_lcls_tools_data_from_file(fname)

    resolution = data["resolution"]
    images = data["images"]

    # process images by centering, cropping, and normalizing
    results = process_images(
        images,
        resolution,
        crop=True,
        center=True,
        n_stds=n_stds,
        max_pixels=max_pixels,
        median_filter_size=3,
        threshold_multiplier=threshold_multiplier,
    )
    resolution = results["pixel_size"]

    if os.path.splitext(fname)[-1] == ".mat":
        final_images = np.mean(results["images"], axis=0)
    else:
        final_images = results["images"]

    final_images = np.clip(final_images - 0.00001, 0.0, None)
    save_name = os.path.split(fname)[-1].split(".")[0] + "_gpsr_prediction"

    # subsample based on process images
    print(f"subsample indicies {results['subsample_idx']}")
    data["quad_strengths"] = data["quad_strengths"][results["subsample_idx"]]
    data["rmat"] = data["rmat"][results["subsample_idx"]]

    if data_slice is not None:
        final_images = final_images[data_slice]
        data["quad_strengths"] = data["quad_strengths"][data_slice]
        data["rmat"] = data["rmat"][data_slice]

    print(f"Final image shape {final_images.shape}")

    return gpsr_fit_quad_scan(
        data["quad_strengths"],
        final_images,
        data["energy"],
        data["rmat"],
        resolution,
        design_twiss=data["design_twiss"],
        visualize=visualize,
        save_location=save_location,
        save_name=save_name,
        **kwargs,
    )


def gpsr_fit_quad_scan(
    quad_strengths,
    images,
    energy,
    rmat,
    resolution,
    n_epochs=500,
    beam_fraction=1.0,
    output_scale=1e-4,
    n_layers=2,
    layer_width=20,
    n_particles=10000,
    design_twiss=None,
    visualize=False,
    save_location=None,
    save_name=None,
):
    """
    Basic method for using GPSR to fit quadrupole scan data.

    This method uses a transformer neural network to fit the quadrupole scan data and
    extract the relevant beam parameters for online control.

    The following hyperparameters can have a significant impact on the reconstruction speed and quality:
    - n_epochs: Number of training epochs. Increasing the number of epochs can improve the fit but
        also increases computation time. See diagnostic plots to verify the convergence.
    - output_scale: Scale of the output beam distribution (should approximate the size of the beam
        in phase space).
    - n_layers: Number of layers in the transformer neural network. Increasing the number of layers can
        (exponentially) improve the model's capacity to learn complex patterns but also increases the
        number of epochs required for training.
    - layer_width: Width of the layers in the transformer neural network. Increasing the layer width can
        (linearly) improve the model's ability to learn complex patterns but also increases the number of
        epochs required for training.
    - n_particles: Number of macroparticles to track. Increasing the number of particles (linearly) improves
        improve the model's ability to learn complex patterns but increases computation time.

    Make sure to visualize and monitor diagnostic plots to verify the convergence.

    Parameters
    ----------
    quad_strengths: list
        Geometric focusing strengths of quadrupole for each image in m^(-2)
    images: np.ndarray
        Array of images to fit (shape: num_images x height x width)
    energy: float
        Energy of the beam in eV
    rmat: np.ndarray
        6D transfer matrix (cheetah coordinate system) from the reconstruction location to the
        diagnostic screen
    resolution: float
        Pixel size of the diagnostic screen in meters.
    n_epochs: int, optional
        Number of training epochs.
    output_scale: float, optional
        Scale of the output beam distribution (should approximate the size of the beam in phase space).
    n_layers: int, optional
        Number of layers in the transformer neural network.
    layer_width: int, optional
        Width of the layers in the transformer neural network.
    n_particles: int, optional
        Number of macroparticles to track.
    beam_fraction: float, optional
        Fraction of the beam to use for measuring the rms parameters (between 0 and 1).
        Defaults to 1.0.
    design_twiss: list, optional
        The design Twiss parameters. Should have the form [beta_x, alpha_x, beta_y, alpha_y].
    visualize: bool, optional
        Whether to visualize diagnostic plots.
    save_location: str, optional
        Location to save diagnostic plots.
    save_name: str, optional
        Name to use for saving the diagnostic plots.

    Returns
    -------
    dict
        A dictionary containing the results of the fitting process.
        The dictionary has the following elements
        - norm_emit_x: Normalized emittance along x in m.rad
        - norm_emit_y: Normalized emittance along y in m.rad
        - beta_x: Beta function along x in m
        - beta_y: Beta function along y in m
        - alpha_x: Alpha function along x
        - alpha_y: Alpha function along y
        - screen_distribution: The distribution of the beam at the screen for each
            quadrupole focusing strength
        - twiss_at_screen: The Twiss parameters at the screen for each quadrupole focusing strength
        - rms_sizes: The RMS sizes of the beam at the screen for each quadrupole focusing strength
        - sigma_matrix: The covariance matrix of the reconstructed beam
        - bmag: The bmag matching parameter for each quadrupole focusing strength
        - reconstructed_distribution: The beam distribution at the reconstruction location.
        - fractional_distribution: The fractional distribution of the beam at the reconstruction location.
        - gpsr_model: The GPSR model used for the fitting process.
        - training_dataset: The training dataset used for the fitting process.
        - prediction_dataset: The predicted dataset from the reconstruction.

    """

    # create cheetah screen diagnostic
    screen = Screen(
        name="screen",
        resolution=images.shape[1:],
        pixel_size=torch.ones(2) * resolution,
        method="kde",
        kde_bandwidth=torch.tensor(resolution, dtype=torch.float32) / 2.0,
        is_active=True,
    )

    # combine into dataset
    train_dset = QuadScanDataset(
        torch.tensor(quad_strengths, dtype=torch.float32).unsqueeze(-1),
        (torch.tensor(images, dtype=torch.float32),),
        screen,
    )

    learning_rate = 1e-2  # learning rate of the optimizer

    # create training model
    R = torch.eye(7).repeat(len(rmat), 1, 1)
    R[:, :6, :6] = rmat

    gpsr_lattice = RMatLattice(R.to(dtype=torch.float32), screen)

    p0c = torch.tensor(energy).to(dtype=torch.float32)
    gpsr_model = GPSR(
        NNParticleBeamGenerator(
            n_particles,
            p0c,
            transformer=NNTransform(
                n_hidden=n_layers,
                width=layer_width,
                output_scale=output_scale,
                phase_space_dim=4,
                activation=CustomLeakyReLU(),
            ),
            n_dim=4,
        ),
        gpsr_lattice,
    )
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=100)

    litgpsr = LitGPSR(gpsr_model, lr=learning_rate)

    # create a pytorch lightning trainer
    cb = MetricTracker()
    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=n_epochs,
        callbacks=[cb],
        accelerator="gpu",
        devices=1,
    )

    # run the training
    start = time.time()
    trainer.fit(model=litgpsr, train_dataloaders=train_loader)
    print(time.time() - start)

    # get the reconstructed beam distribution
    reconstructed_beam = litgpsr.gpsr_model.beam_generator()

    # predict the measurements to compare with training data
    pred = gpsr_model(train_dset.parameters)[0].detach()
    pred_dset = QuadScanDataset(train_dset.parameters, (pred,), screen)

    # grab a fraction of the beam for emittance / twiss calculations
    # if the cholesky factorization fails then return the full beam
    if beam_fraction < 1.0:
        print(f"getting beam fraction {beam_fraction}")
        fractional_beam = get_beam_fraction(reconstructed_beam, beam_fraction)

    else:
        fractional_beam = reconstructed_beam

    # get the reconstructed beam emittances and twiss parameters
    results = get_beam_stats(fractional_beam, gpsr_model, design_twiss)

    if visualize or save_location is not None:
        # compare the predicted measurements with the training data
        fig, ax = plt.subplots(3, len(quad_strengths), sharex="all", sharey="all")

        # plot training data
        train_dset.plot_data(
            overlay_data=pred_dset,
            overlay_kwargs={"levels": [0.01, 0.25, 0.75, 0.9], "cmap": "Greys"},
            filter_size=0,
            ax=ax[0],
            add_labels=False,
        )

        i = 1
        for ele in [train_dset, pred_dset]:
            ele.plot_data(ax=ax[i], add_labels=False)
            i += 1

        fig.set_size_inches(10, 5)
        fig.tight_layout()

        # plot loss curve
        fig2, ax2 = plt.subplots()
        ax2.semilogy(cb.training_loss)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Training Loss")

        # plot distribution
        fig3, ax = fractional_beam.plot_distribution(dimensions=("x", "px", "y", "py"))

        save_name = save_name or "gpsr_training"
        if save_location is not None:
            fig.savefig(os.path.join(save_location, save_name) + ".png")
            fig2.savefig(os.path.join(save_location, "loss_" + save_name) + ".png")
            fig3.savefig(os.path.join(save_location, "dist_" + save_name) + ".png")

    results.update(
        {
            "reconstructed_distribution": reconstructed_beam,
            "fractional_distribution": fractional_beam,
            "gpsr_model": gpsr_model,
            "prediction_dataset": pred_dset,
            "training_dataset": train_dset,
        }
    )

    return results
