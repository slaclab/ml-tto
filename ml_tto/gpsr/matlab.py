import numpy as np
import torch
import os

from gpsr.data_processing import process_images

from ml_tto.gpsr import matlab_parser
from ml_tto.gpsr.quadrupole_scan_fitting import gpsr_fit_quad_scan


def get_matlab_data(fname):
    """
    Load data from LCLS/LCLS-II Matlab emittance measurement.

    Parameters
    ----------
    fname: str
        The path to the .mat file.

    Returns
    -------
    dict
        A dictionary containing the loaded data.
    """
    data = matlab_parser.loadmat(fname)

    # get parameters
    quad_strengths = data["data"]["quadVal"]
    energy = data["data"]["energy"] * 1e9
    rmat = torch.tensor(np.array(data["data"]["rMatrix"]))
    resolution = data["data"]["dataList"][0]["res"][0] * 1e-6

    # get images from matlab and pre-process the images
    images = []
    for ele in data["data"]["dataList"]:
        images += [np.array(ele["img"])]
    images = np.stack(images).transpose(1, 0, -1, -2)

    # get the design twiss
    design_twiss = [
        *data["data"]["twiss0"].T[0][1:],
        *data["data"]["twiss0"].T[1][1:],
    ]

    return {
        "quad_strengths": quad_strengths,
        "energy": energy,
        "rmat": rmat,
        "resolution": resolution,
        "images": images,
        "design_twiss": design_twiss,
    }


def gpsr_fit_matlab(
    fname: str,
    n_epochs: int = 500,
    beam_fraction: float = 1.0,
    pool_size: int = 1,
    save_location: str = None,
    visualize: bool = False,
):
    """
    Use GPSR to fit the beam distribution from Matlab emittance measurements taken at LCLS/LCLS-II.

    Parameters
    ----------
    fname: str
        The path to the .mat file.
    n_epochs: int
        The number of training epochs.
    pool_size: int, optional
        Size of pooling layer to compress images to smaller sizes for speeding up GPSR fitting.
    save_location: str, optional
        The location to save diagnostic plots.
    beam_fraction: float, optional
        The core fraction of the beam to use for fitting. See `get_core_beam` for more information.
    visualize: bool, optional
        Whether to visualize the diagnostic plots.

    """

    matlab_data = get_matlab_data(fname)
    resolution = matlab_data["resolution"]
    images = matlab_data["images"]

    # process images by centering, cropping, and normalizing
    results = process_images(
        images,
        resolution * 1e6,
        crop=True,
        n_stds=4,
        pool_size=pool_size,
    )
    resolution *= pool_size
    final_images = np.mean(results["images"], axis=1)

    save_name = os.path.split(fname)[-1].split(".")[0] + "_gpsr_prediction"

    return gpsr_fit_quad_scan(
        matlab_data["quad_strengths"],
        final_images,
        matlab_data["energy"],
        matlab_data["rmat"],
        resolution,
        n_epochs,
        beam_fraction,
        design_twiss=matlab_data["design_twiss"],
        visualize=visualize,
        save_location=save_location,
        save_name=save_name,
    )
