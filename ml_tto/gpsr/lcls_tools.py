import h5py
import torch
import numpy as np
from lcls_tools.common.data.model_general_calcs import bdes_to_kmod, build_quad_rmat


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

    # depending on the context `info["quadrupole_pv_values"]` could be a list or a dict
    if isinstance(info["quadrupole_pv_values"], dict):
        quad_pv_values = np.array(
            [
                ele
                for ele in list(info["quadrupole_pv_values"]["0"])
                + list(info["quadrupole_pv_values"]["1"])
            ]
        )
    elif isinstance(info["quadrupole_pv_values"], list):
        quad_pv_values = np.array(
            [
                ele
                for ele in list(info["quadrupole_pv_values"][0])
                + list(info["quadrupole_pv_values"][1])
            ]
        )

    idx_sort = np.argsort(quad_pv_values)
    quad_pv_values = quad_pv_values[idx_sort]

    quad_focusing_strengths = bdes_to_kmod(
        e_tot=energy, effective_length=l_eff, bdes=quad_pv_values
    )

    images = np.array(
        [
            metadata["image_data"][str(ele)]["processed_images"][:]
            for ele in quad_pv_values
        ]
    ).squeeze()

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
        "images": images,
        "design_twiss": design_twiss,
    }
