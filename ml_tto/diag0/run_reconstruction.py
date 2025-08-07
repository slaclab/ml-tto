import argparse
import sys
import logging
import yaml
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("watcher.log"), logging.StreamHandler(sys.stdout)],
)

import os
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd

from cheetah.accelerator import Segment

import lightning as L

from gpsr.modeling import GPSR
from gpsr.train import LitGPSR
from gpsr.beams import NNParticleBeamGenerator, NNTransform
from gpsr.modeling import GenericGPSRLattice

NUM_EPOCHS = 1000
SAVE_INTERVAL = 100

def bdes_to_kmod(e_tot=None, effective_length=None, bdes=None, tao=None, element=None):
    """Returns K in 1/m^2 given BDES
    Need to privide either particle energy e_tot and quad effective_length
    or element name and tao object"""
    if e_tot is not None and effective_length is not None and bdes is not None:
        bp = e_tot / 1e9 / 299.792458 * 1e4  # kG m
    elif element is not None and tao is not None:
        ele = tao.ele_gen_attribs(element)
        bp = ele["E_TOT"] / 1e9 / 299.792458 * 1e4  # kG m
        effective_length = ele["L"]
    return bdes / effective_length / bp  # kG / m / kG m = 1/m^2


def visualize_images(images, ax=None, **kwargs):
    # visualize the first shot for each tcav/quad strength
    if ax is None:
        fig, ax = plt.subplots(1, images.shape[0], sharex="all", sharey="all")
        fig.set_size_inches(20, 2)
    else:
        fig = None
    for jj in range(images.shape[0]):
        ax[jj].imshow(images[jj], **kwargs)

    return fig, ax


class DIAG0GPSRLattice(GenericGPSRLattice):
    def set_lattice_parameters(self, settings: torch.Tensor):
        """
        Sets the parameters of variable elements in the segment.

        Args:
            settings: A tensor containing the new parameter values for the variable elements.
        """
        for i, element in enumerate(self.variable_elements, 0):
            setattr(element[0], element[1], settings[..., i])

        self.screen_flags = settings[..., -1]

    def track_and_observe(self, beam):
        """
        Tracks a beam through the segment and collects observations from designated elements.

        Args:
            beam: The beam object to be tracked.

        Returns:
            A tuple of tensors representing the observations from the observable elements.
        """

        # Compute the merged transfer maps for the segment
        merged_segment = self.segment.transfer_maps_merged(beam)

        # Apply the merged segment transformations to the beam
        merged_segment(beam)

        # Collect observations from the observable elements

        observations = tuple([element.reading for element in self.observable_elements])

        return observations[0][:, 0], observations[1][:, 1]


def run_reconstruction(dataset, diag0_lattice_file, dump_location, hyper_params):
    """
    Runs the 6D reconstruction on DIAG0 using the provided dataset and lattice file.

    Parameters:
        dataset (Dataset): The dataset containing the beam parameters and observations.
        diag0_lattice_file (str): Path to the DIAG0 lattice file in JSON format.
        dump_location (str): Directory where the results will be saved.

    """
    # import cheetah lattice
    diag0 = Segment.from_lattice_json(diag0_lattice_file)
    reconstruction_lattice = diag0.subcell(start="bpmdg000")

    # maybe a better structure can be used here where keys automatically line up with the indices of the dataset
    # import data
    p0c = dataset.metadata["beam_energy_GeV"] * 1e9
    dataset_keys = [f"qdg0{i:0>2}" for i in range(1, 12)] + ["TCAV", "screen"]

    dataset.parameters = dataset.parameters
    dataset_keys = dataset_keys

    # scale the strengths of parameters to real units
    for i, name in enumerate(dataset_keys):
        if "qdg" in name:
            dataset.parameters[..., i] = bdes_to_kmod(
                e_tot=p0c,
                effective_length=getattr(diag0, name).length,
                bdes=dataset.parameters[..., i],
            )
        elif "TCAV" in name:
            # scale the TCAV voltage
            dataset.parameters[..., i] = dataset.parameters[..., i]
        else:
            pass

    # convert to float
    dataset.parameters = dataset.parameters.float()

    dataset.parameters = dataset.parameters
    dataset.observations = (dataset.observations[0], dataset.observations[1])

    # set OTR screen resolution to match image shapes
    reconstruction_lattice.otrdg02.resolution = dataset.observations[0].shape[-2:][::-1]
    reconstruction_lattice.otrdg04.resolution = dataset.observations[1].shape[-2:][::-1]

    # scale pixel sizes
    for ele in ["otrdg02", "otrdg04"]:
        pool_size = dataset.metadata[ele + "_pool_size"]
        if pool_size is not None:
            getattr(reconstruction_lattice, ele).pixel_size = (
                reconstruction_lattice.otrdg02.pixel_size * pool_size
            )

    reconstruction_lattice.otrdg02.kde_bandwidth = (
        reconstruction_lattice.otrdg02.pixel_size[0]
    )
    reconstruction_lattice.otrdg04.kde_bandwidth = (
        reconstruction_lattice.otrdg04.pixel_size[0]
    )

    # create GPSR Lattice class
    variable_elements = [
        (getattr(reconstruction_lattice, name), "k1")
        for name in dataset_keys
        if "qdg" in name
    ]
    variable_elements.append((reconstruction_lattice.tcxdg0, "voltage"))
    observable_elements = [
        reconstruction_lattice.otrdg02,
        reconstruction_lattice.otrdg04,
    ]

    gpsr_lattice = DIAG0GPSRLattice(
        segment=reconstruction_lattice,
        variable_elements=variable_elements,
        observable_elements=observable_elements,
    )

    # set up reconstruction
    # hparams -> 
    n_hidden = int(hyper_params.get("n_hidden", 2))
    width = int(hyper_params.get("width", 25))
    lr = float(hyper_params.get("lr", 1e-2))
    dropout= float(hyper_params.get("dropout", 0.0))
    batch_size = int(hyper_params.get("batch_size", 100))
    output_scale = float(hyper_params.get("output_scale", 1e-4))

    gpsr_model = GPSR(
        NNParticleBeamGenerator(
            n_particles=50000, energy=p0c, transformer=NNTransform(n_hidden=n_hidden,
                width=width, dropout=dropout, output_scale=output_scale)
        ),
        gpsr_lattice,
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    litgpsr = LitGPSR(gpsr_model, lr=lr)
    

    ####

    logger = L.pytorch.loggers.CSVLogger(
        os.path.join(dump_location, "logs"),
    )
    logging.info(f"[Watcher] Using logger directory: {logger.log_dir}")
    
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        save_weights_only=True,
        every_n_epochs=SAVE_INTERVAL,
        save_top_k=-1,
        dirpath=dump_location,
        filename="model-{epoch:03d}",
    )

    torch.set_float32_matmul_precision("medium")
    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=NUM_EPOCHS,
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback],
    )
    start = time.time()
    trainer.fit(model=litgpsr, train_dataloaders=train_loader)
    print("reconstruction time: " + str(time.time() - start))

    # Visualize training loss
    try:
        metrics_path = os.path.join(logger.log_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path)
            fig_loss, ax_loss = plt.subplots()
            metrics.plot(y="loss_epoch", logy=True, ax=ax_loss)
            ax_loss.set_title("Training Loss (log scale)")
            ax_loss.set_ylabel("Loss")
            fig_loss.tight_layout()
            fig_loss.savefig(os.path.join(dump_location, "loss_plot.png"))
            plt.close(fig_loss)
        else:
            logging.warning(f"[Watcher] metrics.csv not found at {metrics_path}")
    except Exception as e:
        logging.error(f"[Watcher] Failed to generate loss plot: {e}")

    # get reconstructed distribution
    reconstructed_beam = litgpsr.gpsr_model.beam_generator()
    fig, ax = reconstructed_beam.plot_distribution()

    # save files to dump location
    fig.savefig(os.path.join(dump_location, "6d_reconstruction.png"))
    torch.save(reconstructed_beam, os.path.join(dump_location, "reconstructed_beam.pt"))

    # compare predictions to data
    gt_observations = dataset.observations
    pred = gpsr_model(dataset.parameters)
    for j in range(2):
        fig, ax = plt.subplots(2, len(gt_observations[j]), sharex="all", sharey="all")
        for i, ele in enumerate([gt_observations, pred]):
            visualize_images(ele[j].detach(), ax=ax[i], origin="lower", aspect="auto")

        fig.set_size_inches(10, 3)
        fig.tight_layout()
        fig.savefig(os.path.join(dump_location, f"otrdg0{2 * (j + 1)}_predictions.png"))

    # get reconstructions from checkpoints
    for epoch in range(SAVE_INTERVAL - 1, NUM_EPOCHS, SAVE_INTERVAL):
        # load weights from checkpoint

        checkpoint = torch.load(f"{dump_location}/model-epoch={epoch:03d}.ckpt")
        litgpsr.load_state_dict(checkpoint["state_dict"])

        # get reconstructed distribution
        reconstructed_beam = litgpsr.gpsr_model.beam_generator()
        fig, ax = reconstructed_beam.plot_distribution()

        # save files to dump location
        fig.savefig(os.path.join(dump_location, f"6d_reconstruction-{epoch:03d}.png"))
        plt.close(fig)
        torch.save(reconstructed_beam, os.path.join(dump_location, f"reconstructed_beam-{epoch:03d}.pt"))

        # compare predictions to data
        gt_observations = dataset.observations
        pred = gpsr_model(dataset.parameters)
        for j in range(2):
            fig, ax = plt.subplots(2, len(gt_observations[j]), sharex="all", sharey="all")
            for i, ele in enumerate([gt_observations, pred]):
                visualize_images(ele[j].detach(), ax=ax[i], origin="lower", aspect="auto")

            fig.set_size_inches(10, 3)
            fig.tight_layout()
            fig.savefig(os.path.join(dump_location, f"otrdg0{2 * (j + 1)}_predictions-{epoch:03d}.png"))
            plt.close(fig)

def main():
    logging.info("Entering code")
    parser = argparse.ArgumentParser(description="6D reconstruction for DIAG0")
    parser.add_argument(
        "--dump_location", type=str, help="path to dump directory", required=True
    )
    parser.add_argument(
        "--diag0_lattice_file",
        type=str,
        help="path to diag0 lattice file",
        required=True,
    )
    parser.add_argument("--processed_data", type=str, help="processed data")
    parser.add_argument("--hyper_params",type=str, help="hyper params yaml" )
    args = parser.parse_args()

    dataset = torch.load(args.processed_data, weights_only=False)
    try:
        with open(args.hyper_params) as f:
            hyper_params= yaml.safe_load(f)
    except Exception as e:
        print(f'Exception {e} when loading in hyper params, running in default mode')
        #logging.warning(f'Exception {e} when loading in hyper params, running in default mode')
        hyper_params = {}

    print(hyper_params)
    new_dir = args.dump_location + 'nhidden_' + str(hyper_params.get("n_hidden",3)) + '_width_' + str(hyper_params.get("hidden_width",25))
    print(new_dir)
    os.mkdir(new_dir)
    # do reconstruction
    run_reconstruction(dataset, args.diag0_lattice_file, new_dir, hyper_params)


if __name__ == "__main__":
    main()
