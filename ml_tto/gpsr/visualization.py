import atexit
import io
import os
import shutil
import tempfile

import lightning as L
import matplotlib.pyplot as plt
import torch
from gpsr.datasets import QuadScanDataset
from gpsr.train import LitGPSR
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm



def visualize_quad_scan_result(
    quad_strengths, train_dset, pred_dset, cb, stats, fractional_beam
):
    fig1 = plot_measurement_comparison(quad_strengths, train_dset, pred_dset)
    fig2 = plot_training_loss(cb)
    fig3 = plot_4d_distribution(fractional_beam, stats)
    return fig1, fig2, fig3


def plot_measurement_comparison(quad_strengths, train_dset, pred_dset):
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
    return fig


def plot_training_loss(cb):
    # plot loss curve
    fig, ax = plt.subplots()
    ax.semilogy(cb.training_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    return fig


def plot_4d_distribution(fractional_beam, stats):
    # plot distribution
    fig, ax = fractional_beam.plot_distribution(dimensions=("x", "px", "y", "py"))

    # add distribution statistics to plot distribution in the top left corner
    # TODO: compute and display twiss parameters at nominal quad strength
    labels = {
        "norm_emittance_x": r"$\epsilon_x$",
        "norm_emittance_y": r"$\epsilon_y$",
        # "beta_x": r"$\beta_x$",
        # "beta_y": r"$\beta_y$",
        # "alpha_x": r"$\alpha_x$",
        # "alpha_y": r"$\alpha_y$",
        "halo_x": r"$\text{Halo}_x$",
        "halo_y": r"$\text{Halo}_y$",
    }
    units = {
        "norm_emittance_x": r"$[mm \cdot mrad]$",
        "norm_emittance_y": r"$[mm \cdot mrad]$",
        "beta_x": r"$[m]$",
        "beta_y": r"$[m]$",
        "alpha_x": r"[]",
        "alpha_y": r"[]",
        "halo_x": r"[]",
        "halo_y": r"[]",
    }

    info_str = ""
    for name, lbl in labels.items():
        try:
            info_str += f"{lbl}: {stats[name]:<.4f} {units[name]}\n"
        except KeyError:
            pass

    fig.text(
        0.6,
        0.9,
        info_str,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=15,
    )

    return fig


def combine_images_with_title(img1, img2, title, title_height=50, font_size=30):
    # scale second image proportionally to match first image's height
    w1, h1 = img1.size
    w2, h2 = img2.size
    new_w2 = round(w2 * (h1 / h2))
    img2 = img2.resize((new_w2, h1), Image.Resampling.LANCZOS)

    # prepare title banner
    total_width = w1 + new_w2
    title_img = Image.new("RGB", (total_width, title_height), "white")

    draw = ImageDraw.Draw(title_img)
    font_properties = font_manager.FontProperties(family=["sans-serif"])
    font_path = font_manager.findfont(font_properties)
    font = ImageFont.truetype(font_path, font_size)

    # draw centered title
    _, _, text_w, text_h = draw.textbbox((0, 0), title, font=font)
    draw.text(((total_width - text_w) / 2, (title_height - text_h) / 2),
              title, font=font, fill="black")

    # combine everything
    total_height = title_height + h1
    combined = Image.new("RGB", (total_width, total_height), "white")
    combined.paste(title_img, (0, 0))
    combined.paste(img1, (0, title_height))
    combined.paste(img2, (w1, title_height))

    return combined


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)


def save_gif(frames, frame_delay, loop_delay, gif_path):
    durations = [frame_delay * 1000] * (len(frames) - 1) + [loop_delay * 1000]
    frames[0].save(gif_path,
               save_all=True,
               append_images=frames[1:],
               duration=durations,    # duration per frame in ms
               loop=0)                # 0 means loop forever


def animate_gpsr_file(
    fname: str,
    data_slice: slice = None,
    save_location: str = None,
    save_name: str = None,
    max_pixels: int = 1e5,
    n_stds: int = 5,
    threshold_multiplier=1.2,
    n_epochs=500,
    frame_delay=0.25,
    loop_delay=5.0,
    **kwargs,
):
    from ml_tto.gpsr.quadrupole_scan_fitting import gpsr_fit_file

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

    # fit beam distribution
    results = gpsr_fit_file(
        fname=fname,
        data_slice=data_slice,
        save_location=None,
        visualize=False,
        max_pixels=max_pixels,
        n_stds=n_stds,
        threshold_multiplier=threshold_multiplier,
        callbacks=[checkpoint_cb],
        n_epochs=n_epochs,
        **kwargs,
    )

    reconstructed_beam = results["reconstructed_distribution"]
    gpsr_model = results["gpsr_model"]
    train_dset = results["training_dataset"]

    # generate reconstruction animation
    gif_frames = []
    dimensions = ["x", "px", "y", "py"]
    bin_ranges = None

    # determine bin ranges from last epoch
    # source: https://github.com/desy-ml/cheetah/blob/200ef469b9ac776ea17e818a7022e2b9d306d4ca/cheetah/particles/particle_beam.py#L1408
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

    print("generating distribution and measurement plots")
    litgpsr = LitGPSR(gpsr_model)
    screen = train_dset.screen
    quad_strengths = train_dset.parameters.squeeze(-1)
    for epoch in tqdm(range(n_epochs)):
        # load weights from checkpoint
        checkpoint_path = checkpoint_cb.format_checkpoint_name({"epoch": epoch})
        checkpoint = torch.load(checkpoint_path)
        litgpsr.load_state_dict(checkpoint["state_dict"])
        litgpsr.to("cuda")

        # generate distribution plot
        reconstructed_beam = litgpsr.gpsr_model.beam_generator()
        fig1, _ = reconstructed_beam.plot_distribution(dimensions=dimensions, bin_ranges=bin_ranges)
        img1 = fig_to_png(fig1)
        plt.close(fig1)

        # generate measurement plot
        pred = litgpsr.gpsr_model(train_dset.parameters)[0].detach()
        pred_dset = QuadScanDataset(train_dset.parameters, (pred.cpu(),), screen)
        fig2 = plot_measurement_comparison(quad_strengths, train_dset, pred_dset)
        img2 = fig_to_png(fig2)
        plt.close(fig2)

        # place plots side by side
        title = f"Reconstructed distribution and predicted measurements (epoch {epoch + 1})"
        combined = combine_images_with_title(img1, img2, title)
        gif_frames.append(combined)

        # delete checkpoint
        os.remove(checkpoint_path)

    # save frames as gif
    save_name = save_name or os.path.split(fname)[-1].split(".")[0] + "_gpsr_prediction"
    if save_location is not None:
        print("saving animation to gif")
        gif_path = os.path.join(save_location, save_name + "_dist_pred") + ".gif"
        save_gif(gif_frames, frame_delay, loop_delay, gif_path)
