import atexit
import io
import shutil
import tempfile
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from gpsr.modeling import GPSR
from gpsr.train import LitGPSR
from gpsr.beams import NNParticleBeamGenerator, NNTransform
from gpsr.datasets import QuadScanDataset
from gpsr.data_processing import process_images
from gpsr.analysis import get_beam_fraction
from cheetah.accelerator import Screen
from cheetah.particles.particle_beam import ParticleBeam
from cheetah.particles.species import Species
import lightning as L

import time

from ml_tto.gpsr.utils import (
    get_beam_stats,
    RMatLattice,
    CustomLeakyReLU,
    MetricTracker,
)
from ml_tto.gpsr.visualization import visualize_quad_scan_result


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
    animate=False,
    frame_delay=0.25,
    loop_delay=5.0,
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
    animate: bool, optional
        Whether to save diagnostic animations.
    frame_delay: float, optional
        Delay between frames of each animation in seconds.
    loop_delay: float, optional
        Delay between loops of each animation in seconds.

    Returns
    -------
    dict
        A dictionary containing the results of the fitting process.
        The dictionary has the following elements
        - norm_emit_x: Normalized emittance along x in mm.mrad
        - norm_emit_y: Normalized emittance along y in mm.mrad
        - emittance_x: Geometric emittance along x in mm.mrad
        - emittance_y: Geometric emittance along y in mm.mrad
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

    # create callbacks
    metric_cb = MetricTracker()
    if animate:
        # temporarily save checkpoints for reconstruction
        checkpoint_dir = tempfile.mkdtemp(dir=os.getcwd())
        atexit.register(shutil.rmtree, checkpoint_dir) # clean up on exit, even if exception is raised
        checkpoint_cb = L.pytorch.callbacks.ModelCheckpoint(
            save_weights_only=True,
            every_n_epochs=1,
            save_top_k=-1,
            dirpath=checkpoint_dir,
            filename="model-{epoch:03d}",
        )
        callbacks = [metric_cb, checkpoint_cb]
    else:
        callbacks = [metric_cb]

    # create a pytorch lightning trainer
    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=n_epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
    )

    # run the training
    start = time.time()
    trainer.fit(model=litgpsr, train_dataloaders=train_loader)
    print("Runtime:", time.time() - start)

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
    if visualize:
        fig1, fig2, fig3 = visualize_quad_scan_result(
            quad_strengths, train_dset, pred_dset, metric_cb, results, fractional_beam
        )

        save_name = save_name or "gpsr_training"
        if save_location is not None:
            fig1.savefig(os.path.join(save_location, save_name + "_pred") + ".png")
            fig2.savefig(os.path.join(save_location, save_name + "_loss") + ".png")
            fig3.savefig(os.path.join(save_location, save_name + "_dist") + ".png")

    results.update(
        {
            "reconstructed_distribution": reconstructed_beam,
            "fractional_distribution": fractional_beam,
            "gpsr_model": gpsr_model,
            "prediction_dataset": pred_dset,
            "training_dataset": train_dset,
        }
    )

    if animate:
        # generate reconstruction animation
        beam_frames = []
        pred_frames = []
        dimensions = ["x", "px", "y", "py"]
        bin_ranges = None

        # pass 1: read checkpoints and save reconstructed beams and predicted measurements
        print('generating reconstructed beams and predicted measurements')
        for epoch in tqdm(range(n_epochs)):
            # load weights from checkpoint
            checkpoint_path = checkpoint_cb.format_checkpoint_name({"epoch": epoch})
            checkpoint = torch.load(checkpoint_path)
            litgpsr.load_state_dict(checkpoint["state_dict"])
            litgpsr.to("cuda")

            # perform 4d reconstruction
            reconstructed_beam = litgpsr.gpsr_model.beam_generator()

            # predict the measurements to compare with training data
            pred = litgpsr.gpsr_model(train_dset.parameters)[0].detach()
            pred_dset = QuadScanDataset(train_dset.parameters, (pred.cpu(),), screen)

            # save reconstruction and delete checkpoint
            beam_data_fname = f"beam-{epoch:03d}.pt"
            beam_data_path = os.path.join(checkpoint_dir, beam_data_fname)
            torch.save(reconstructed_beam, beam_data_path)
            pred_data_fname = f"pred-{epoch:03d}.pt"
            pred_data_path = os.path.join(checkpoint_dir, pred_data_fname)
            torch.save(pred_dset, pred_data_path)
            os.remove(checkpoint_path)

        # determine bin ranges from last epoch
        full_tensor = (
            torch.stack([getattr(reconstructed_beam, dimension) for dimension in dimensions], dim=-2)
            .cpu()
            .detach()
            .numpy()
        )
        bin_ranges = [
            (
                full_tensor[i, :].min()
                - (full_tensor[i, :].max() - full_tensor[i, :].min()) / 10,
                full_tensor[i, :].max()
                + (full_tensor[i, :].max() - full_tensor[i, :].min()) / 10,
            )
            for i in range(full_tensor.shape[-2])
        ]

        # pass 2: read and plot reconstructed beams and predicted measurements
        print('generating plots')
        for epoch in tqdm(range(n_epochs)):
            # load reconstructed beam
            beam_data_fname = f"beam-{epoch:03d}.pt"
            beam_data_path = os.path.join(checkpoint_dir, beam_data_fname)
            with torch.serialization.safe_globals([ParticleBeam, Species]):
                reconstructed_beam = torch.load(beam_data_path)

            # generate distribution image
            reconstructed_beam.plot_distribution(dimensions=dimensions, bin_ranges=bin_ranges)
            plt.suptitle(f"4D reconstruction (epoch {epoch + 1})")

            # save frame
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            beam_frames.append(img)

            # load predicted measurements
            pred_data_fname = f"pred-{epoch:03d}.pt"
            pred_data_path = os.path.join(checkpoint_dir, pred_data_fname)
            with torch.serialization.safe_globals([QuadScanDataset, Screen]):
                pred_dset = torch.load(pred_data_path)

            # compare the predicted measurements with the training data
            fig, ax = plt.subplots(
                3,
                len(quad_strengths),
                sharex="all",
                sharey="all",
                gridspec_kw={"hspace": 0.05, "wspace": 0.05},
            )

            # plot training / predicted data
            i = 0
            for ele in [train_dset, pred_dset]:
                ele.plot_data(ax=ax[i], add_labels=False)
                i += 1

            # plot overlay comparison
            train_dset.plot_data(
                overlay_data=pred_dset,
                overlay_kwargs={"levels": [0.01, 0.25, 0.75, 0.9], "cmap": "Greys"},
                filter_size=0,
                ax=ax[2],
                add_labels=False,
            )

            # add labels
            ax[0, 0].text(
                -0.1,
                1.1,
                "$k_1$ (1/m$^2$)",
                va="bottom",
                ha="right",
                transform=ax[0, 0].transAxes,
            )

            label = ["Measured", "Predicted", "Overlay"]
            for kk in range(3):
                ax[kk, 0].text(
                    -1.25,
                    0.5,
                    label[kk],
                    va="center",
                    ha="right",
                    transform=ax[kk, 0].transAxes,
                    rotation="vertical",
                    size="large",
                    weight="bold",
                )

            # set titles for each subplot
            for j in range(len(quad_strengths)):
                ax[0, j].set_title(f"{quad_strengths[j]:.2f}")

            # set axes labels
            for j in range(len(quad_strengths)):
                ax[-1, j].set_xlabel("x [mm]")

            for k in range(3):
                ax[k, 0].set_ylabel("y [mm]")

            fig.set_size_inches(len(quad_strengths), 5)
            plt.suptitle(f"Training vs. predicted data (epoch {epoch + 1})")

            # save frame
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            pred_frames.append(img)

        # save frames as gif
        print('saving animations to gif')
        durations = [frame_delay * 1000] * (len(beam_frames) - 1) + [loop_delay * 1000]
        animation_path = os.path.join(save_location, save_name + "_4d_recon") + ".gif"
        beam_frames[0].save(animation_path,
                   save_all=True,
                   append_images=beam_frames[1:],
                   duration=durations,   # duration per frame in ms
                   loop=0)         # 0 means loop forever

        # save frames as gif
        durations = [frame_delay * 1000] * (len(pred_frames) - 1) + [loop_delay * 1000]
        animation_path = os.path.join(save_location, save_name + "_pred") + ".gif"
        pred_frames[0].save(animation_path,
                   save_all=True,
                   append_images=pred_frames[1:],
                   duration=durations,   # duration per frame in ms
                   loop=0)         # 0 means loop forever

    return results
