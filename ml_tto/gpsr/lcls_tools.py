import h5py
import torch
import numpy as np
from lcls_tools.common.data.model_general_calcs import bdes_to_kmod, build_quad_rmat
from gpsr.data_processing import process_images
from lcls_tools.common.measurements.emittance_measurement import (
    EmittanceMeasurementResult,
)
from ml_tto.gpsr.utils import image_snr


def hdf5_group_to_dict(hdf5_object):
    """
    Recursively converts an h5py.File or h5py.Group object into a Python dictionary.
    """
    data_dict = {}
    for key, item in hdf5_object.items():
        if isinstance(item, h5py.Group):
            data_dict[key] = hdf5_group_to_dict(item)  # Recursively call for subgroups
        elif isinstance(item, h5py.Dataset):
            data_dict[key] = item[()]  # Read dataset content
    return data_dict


def get_lcls_tools_data_from_file(fname: str):
    with h5py.File(fname, "r") as f:
        info = hdf5_group_to_dict(f)
    return get_lcls_tools_data(info)


def get_lcls_tools_data(
    info: dict,
    get_all_data: bool = False,
):
    metadata = info["metadata"]
    energy = metadata["energy"]
    rmat = metadata["rmat"]
    resolution = metadata["resolution"] * 1e-6
    design_twiss = [
        metadata["design_twiss"][ele]
        for ele in ["beta_x", "alpha_x", "beta_y", "alpha_y"]
    ]
    l_eff = metadata["magnet"]["metadata"]["l_eff"]

    if get_all_data:
        # get raw pv values from xopt object
        quad_pv_values = np.array(metadata["scan_values"])
    else:
        # depending on the context `info["quadrupole_pv_values"]` could be a list or a dict
        if isinstance(info["quadrupole_pv_values"], dict):
            quad_pv_values = np.unique(
                np.array(
                    [
                        ele
                        for ele in list(info["quadrupole_pv_values"]["0"])
                        + list(info["quadrupole_pv_values"]["1"])
                    ]
                )
            )
        elif isinstance(info["quadrupole_pv_values"], list):
            quad_pv_values = np.unique(
                np.array(
                    [
                        ele
                        for ele in list(info["quadrupole_pv_values"][0])
                        + list(info["quadrupole_pv_values"][1])
                    ]
                )
            )

    quad_focusing_strengths = bdes_to_kmod(
        e_tot=energy, effective_length=l_eff, bdes=quad_pv_values
    )

    # processed_images = np.array(
    #    [
    #        metadata["image_data"][str(ele)]["processed_images"][:]
    #        for ele in quad_pv_values
    #    ]
    # ).squeeze()
    raw_images = np.array(
        [metadata["image_data"][str(ele)]["raw_images"][:] for ele in quad_pv_values]
    ).squeeze()

    # transpose for proper reconstruction
    # processed_images = np.transpose(processed_images, (0, 2, 1))
    raw_images = np.transpose(raw_images, (0, 2, 1))

    quad_x_rmats = build_quad_rmat(
        np.array(quad_focusing_strengths),
        l_eff,
    )
    total_x_rmats = np.expand_dims(rmat[0], -3) @ quad_x_rmats
    quad_y_rmats = build_quad_rmat(
        -np.array(quad_focusing_strengths),
        l_eff,
    )
    total_y_rmats = np.expand_dims(rmat[1], -3) @ quad_y_rmats

    rmats = torch.eye(6).unsqueeze(0).repeat(len(quad_focusing_strengths), 1, 1)
    rmats[..., :2, :2] = torch.tensor(total_x_rmats)
    rmats[..., 2:4, 2:4] = torch.tensor(total_y_rmats)

    return {
        "quad_strengths": quad_focusing_strengths,
        "quad_pv_values": quad_pv_values,
        "energy": energy,
        "rmat": rmats,
        "resolution": resolution,
        "raw_images": raw_images,
        "design_twiss": design_twiss,
    }


def process_automatic_emittance_measurement_data(data: dict, **kwargs):
    """
    Extract and process data from automatic emittance measurement results.
    """
    resolution = data["resolution"]
    # images = data["images"]
    raw_images = data["raw_images"]
    design_twiss = data["design_twiss"]

    # calculate signal to noise ratio for the raw images
    snr_values = np.array([image_snr(img) for img in raw_images])
    snr_condition = snr_values > min_signal_to_noise_ratio

    print(
        f"{np.count_nonzero(snr_condition)} / {len(snr_condition)} images satisfied the signal to noise limit of {min_signal_to_noise_ratio}"
    )

    raw_images = raw_images[snr_condition]
    quad_strengths = data["quad_strengths"][snr_condition]
    rmat = data["rmat"][snr_condition]
    quad_pv_values = data["quad_pv_values"][snr_condition]
    energy = data["energy"]

    # process images by centering, cropping, and normalizing
    results = process_images(raw_images, resolution, crop=True, center=True, **kwargs)
    resolution = results["pixel_size"]

    print(f"Final image shape {results['images'].shape}")

    # subsample based on process images
    print(f"subsample indicies {results['subsample_idx']}")
    quad_strengths = quad_strengths[results["subsample_idx"]]
    quad_pv_values = quad_pv_values[results["subsample_idx"]]
    rmat = rmat[results["subsample_idx"]]

    return {
        "quad_strengths": quad_strengths,
        "quad_pv_values": quad_pv_values,
        "rmat": rmat,
        "resolution": resolution,
        "images": results["images"],
        "design_twiss": design_twiss,
        "energy": energy,
    }
