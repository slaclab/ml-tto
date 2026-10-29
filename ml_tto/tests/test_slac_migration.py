import importlib
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from slac_db.device import get_all_accessors, get_attribute
from slac_devices.magnet import Magnet
from slac_devices.reader import create_device
from slac_measurements.emittance_measurement import QuadScanEmittanceResult
from slac_measurements.fitting.gaussian import curve
from slac_measurements.image.fit import ImageProjectionFit
from slac_model.model_general_calcs import bdes_to_kmod, build_quad_rmat

from ml_tto.automatic_emittance.automatic_emittance import GPSRMLQuadScanEmittance
from ml_tto.automatic_emittance.image_projection_fit import RecursiveImageProjectionFit
from ml_tto.automatic_emittance.plotting import plot_image_projection_fit
from ml_tto.automatic_emittance.utils import (
    compute_emit_bmag_machine_units,
    emittance_from_h5,
    quad_scan_result_from_dict,
)
from ml_tto.io import H5Saver


@pytest.mark.parametrize("includes_quad", [False, True])
def test_quad_scan_result_schema_and_units(includes_quad, tmp_path):
    energy = 125e6
    length = 0.1
    values = np.linspace(-2, 2, 11)
    strengths = bdes_to_kmod(energy, length, values)
    matrices = np.stack(
        [
            np.array([[1.0, 1.0], [0.0, 1.0]])
            @ build_quad_rmat(sign * strengths, length)
            for sign in [1, -1]
        ]
    )
    beam_matrix = np.array([[0.05, -0.02, 0.03], [0.1, 0.02, 0.04]])
    r11, r12 = matrices[..., 0, 0], matrices[..., 0, 1]
    coefficients = np.stack([r11**2, 2 * r11 * r12, r12**2], axis=-1)
    beam_sizes = np.sqrt(np.einsum("ijk,ik->ij", coefficients, beam_matrix)) * 1e-3
    beam_sizes[0, 1] = np.nan
    beam_sizes[1, 3] = np.nan
    results = compute_emit_bmag_machine_units(
        quad_vals=[values, values],
        beamsizes=beam_sizes,
        q_len=length,
        rmat=matrices
        if includes_quad
        else np.tile([[1.0, 1.0], [0.0, 1.0]], (2, 1, 1)),
        energy=energy,
        twiss_design=np.array([[5.0, 1.0], [3.0, -1.0]]),
        maxiter=100,
        rmat_includes_quad=includes_quad,
    )
    result = QuadScanEmittanceResult(**results, energy=energy)
    geometric = np.sqrt(beam_matrix[:, 0] * beam_matrix[:, 2] - beam_matrix[:, 1] ** 2)
    gamma = energy / 511000
    assert np.allclose(result.geometric_emittance[:, 0], geometric, rtol=1e-4)
    assert np.allclose(
        result.emittance[:, 0], geometric * gamma * (1 - 1 / (2 * gamma**2)), rtol=1e-4
    )
    assert np.allclose(result.beam_matrix, beam_matrix, rtol=1e-4)
    assert result.twiss_at_reconstruction.shape == (2, 1, 3)
    assert all(twiss.shape == (10, 3) for twiss in result.twiss)
    assert all(phase.shape == (10,) for phase in result.phase_advances)
    for dim in range(2):
        finite = np.isfinite(beam_sizes[dim])
        assert np.allclose(result.rms_beamsizes[dim], beam_sizes[dim, finite] * 1e6)
        assert np.array_equal(result.quadrupole_pv_values[dim], values[finite])
    saved = result.model_dump()
    saved["metadata"] = {
        "energy": energy,
        "magnet": {"metadata": {"l_eff": length}},
        "rmat": np.tile([[1.0, 1.0], [0.0, 1.0]], (2, 1, 1)),
        "design_twiss": None,
    }
    for legacy in [False, True]:
        if legacy:
            saved.pop("geometric_emittance")
            saved["emittance"] = result.geometric_emittance
            saved["rms_beamsizes"] = [size * 1e-6 for size in result.rms_beamsizes]
        path = tmp_path / f"result_{legacy}.h5"
        H5Saver().dump(saved, path)
        recalculated = emittance_from_h5(path)
        assert np.allclose(recalculated.geometric_emittance, result.geometric_emittance)
        assert np.allclose(recalculated.emittance, result.emittance)


def test_database_device_definitions():
    magnet = create_device("CQ01")
    assert isinstance(magnet, Magnet)
    assert magnet.name == "CQ01"
    assert magnet.metadata.area == get_attribute("CQ01", "area")
    assert (
        magnet.controls_information.PVs.bctrl.pvname
        == get_all_accessors("CQ01")["bctrl"]
    )


def test_hdf5_roundtrip(tmp_path):
    values = {
        "array": np.arange(6).reshape(2, 3),
        "strings": np.array(["screen", "quadrupole"]),
        "list": [None, {"value": 2.5}, np.array([3, 4])],
        "tuple": (1, "two"),
        "timestamp": datetime(2026, 10, 28, 12, 0),
        "table": pd.DataFrame({"k": [1.0, 2.0], "size": [3.0, 4.0]}, index=[4, 8]),
    }
    path = tmp_path / "result.h5"
    saver = H5Saver()
    saver.dump(values, path)
    result = saver.load(path)
    assert np.array_equal(result["array"], values["array"])
    assert np.array_equal(result["strings"], values["strings"])
    assert result["list"][0] is None
    assert result["list"][1] == {"value": 2.5}
    assert np.array_equal(result["list"][2], [3, 4])
    assert result["tuple"] == (1, "two")
    assert result["timestamp"] == values["timestamp"]
    pd.testing.assert_frame_equal(result["table"], values["table"])


def test_hdf5_unsupported_type_is_explicit(tmp_path):
    with pytest.raises(TypeError, match="Cannot serialize object"):
        H5Saver().dump({"unsupported": object()}, tmp_path / "bad.h5")


@pytest.mark.parametrize(
    "filename",
    [
        "fixtures/test_emittance_result.h5",
        "gpsr/test_automatic_emittance_scan.h5",
    ],
)
def test_legacy_result_upgrade(filename):
    data = H5Saver().load(Path(__file__).parent / filename)
    result = quad_scan_result_from_dict(data)
    assert isinstance(result, QuadScanEmittanceResult)
    assert result.energy == data["metadata"]["energy"]
    assert np.array_equal(result.geometric_emittance, data["emittance"])
    assert result.twiss_at_reconstruction.shape == (2, 1, 3)
    sizes = data["rms_beamsizes"]
    first_sizes = sizes["0"] if isinstance(sizes, dict) else sizes[0]
    assert np.allclose(result.rms_beamsizes[0], first_sizes * 1e6)
    reloaded = quad_scan_result_from_dict(result.model_dump())
    assert np.array_equal(reloaded.emittance, result.emittance)
    assert np.array_equal(reloaded.geometric_emittance, result.geometric_emittance)


@pytest.mark.parametrize(
    "centroid, rms_size",
    [([110, 40], [8, 5]), ([40, 110], [5, 8])],
)
def test_recursive_fit_preserves_original_coordinates(centroid, rms_size):
    x, y = np.indices((180, 160))
    image = 1000 * np.exp(
        -0.5
        * (((x - centroid[0]) / rms_size[0]) ** 2
           + ((y - centroid[1]) / rms_size[1]) ** 2)
    )
    result = RecursiveImageProjectionFit().fit_image(image)
    assert result.image.shape[0] < image.shape[0]
    assert result.image.shape[1] < image.shape[1]
    assert np.allclose(result.centroid, centroid, atol=0.1)
    assert np.allclose(result.rms_size, rms_size, atol=0.2)
    assert np.allclose(np.mean(result.beam_extent, axis=1), result.centroid)


def test_projection_plot_uses_current_fitting_module():
    x, y = np.indices((100, 80))
    image = 1000 * np.exp(-0.5 * (((x - 50) / 8) ** 2 + ((y - 40) / 5) ** 2))
    result = ImageProjectionFit().fit_image(image)
    assert np.allclose(result.centroid, [50, 40], atol=0.1)
    assert np.allclose(result.rms_size, [8, 5], atol=0.2)
    fig, axes = plot_image_projection_fit(result)
    assert len(axes) == 3
    assert len(axes[0].images) == 1
    assert np.array_equal(axes[0].images[0].get_array(), image.T)
    for dim, ax in enumerate(axes[1:]):
        assert [line.get_label() for line in ax.lines] == ["data", "fit"]
        projection = image.sum(axis=1 - dim)
        assert np.array_equal(ax.lines[0].get_ydata(), projection)
        parameters = result.projection_fit_parameters[dim]
        expected_fit = curve(
            np.arange(len(projection)),
            **{key: parameters[key] for key in ("mean", "sigma", "amp", "off")},
        )
        np.testing.assert_allclose(ax.lines[1].get_ydata(), expected_fit)
    plt.close(fig)


def test_gpsr_populates_current_result_schema():
    geometric = np.array([[0.01], [0.02]])
    initial = QuadScanEmittanceResult(
        energy=125e6,
        emittance=geometric * 200,
        geometric_emittance=geometric,
        beam_matrix=np.ones((2, 3)),
        twiss=np.ones((2, 3, 3)),
        twiss_at_reconstruction=np.ones((2, 1, 3)),
        quadrupole_focusing_strengths=[np.arange(3)] * 2,
        quadrupole_pv_values=[np.arange(3)] * 2,
        rms_beamsizes=[np.ones(3)] * 2,
        metadata={},
    )
    processed = {
        "quad_strengths": np.arange(3),
        "quad_pv_values": np.arange(3),
        "images": np.ones((3, 10, 10)),
        "energy": 125e6,
        "rmat": np.tile(np.eye(6), (3, 1, 1)),
        "resolution": 1e-5,
        "design_twiss": None,
    }
    gpsr = {
        "emittance_x": 0.01,
        "emittance_y": 0.02,
        "norm_emittance_x": 2.0,
        "norm_emittance_y": 4.0,
        "beta_x": 5.0,
        "beta_y": 3.0,
        "alpha_x": 1.0,
        "alpha_y": -1.0,
        "twiss": np.ones((2, 3, 3)),
        "bmag": None,
        "rms_beamsizes": np.ones((2, 3)) * 1e-4,
        "beam_matrix": np.ones((2, 3)) * 1e-6,
    }
    module = importlib.import_module("ml_tto.automatic_emittance.automatic_emittance")
    measurement = GPSRMLQuadScanEmittance.model_construct(energy=125e6)
    with (
        patch.object(
            module.MLQuadScanEmittance, "calculate_emittance", return_value=initial
        ),
        patch.object(module, "get_lcls_tools_data", return_value={}),
        patch.object(
            module,
            "process_automatic_emittance_measurement_data",
            return_value=processed,
        ),
        patch.object(module, "gpsr_fit_quad_scan", return_value=gpsr),
    ):
        result, gpsr_result = measurement.calculate_emittance()
    assert gpsr_result is gpsr
    assert np.array_equal(result.geometric_emittance, geometric)
    assert np.array_equal(result.emittance, [[2.0], [4.0]])
    assert np.allclose(result.beam_matrix, 1.0)
    assert np.allclose(result.rms_beamsizes, 100.0)
    assert result.twiss_at_reconstruction.shape == (2, 1, 3)
    assert result.energy == 125e6
