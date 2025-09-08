import matplotlib.pyplot as plt


def visualize_quad_scan_result(
    quad_strengths, train_dset, pred_dset, cb, stats, fractional_beam
):
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

    # plot loss curve
    fig2, ax2 = plt.subplots()
    ax2.semilogy(cb.training_loss)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")

    # plot distribution
    fig3, ax = fractional_beam.plot_distribution(dimensions=("x", "px", "y", "py"))

    # add distribution statistics to plot distribution in the top left corner
    labels = {
        "norm_emittance_x": r"$\epsilon_x$",
        "norm_emittance_y": r"$\epsilon_y$",
        "beta_x": r"$\beta_x$",
        "beta_y": r"$\beta_y$",
        "alpha_x": r"$\alpha_x$",
        "alpha_y": r"$\alpha_y$",
        "halo_x": r"$\text{Halo}_x$",
        "halo_y": r"$\text{Halo}_y$",
    }
    info_str = ""
    for name, lbl in labels.items():
        try:
            if name in ["norm_emittance_x", "norm_emittance_y"]:
                info_str += f"{lbl}: {stats[name] * 1e6:<.4f}\n"
            else:
                info_str += f"{lbl}: {stats[name]:<.4f}\n"
        except KeyError:
            pass

    fig3.text(
        0.7,
        0.9,
        info_str,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=15,
    )

    return fig, fig2, fig3
