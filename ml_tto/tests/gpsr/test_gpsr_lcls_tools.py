import h5py
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# get path to test file
test_file_dir = os.path.dirname(os.path.abspath(__file__))

from ml_tto.gpsr.lcls_tools import (
    get_lcls_tools_data_from_file,
    hdf5_group_to_dict,
    process_automatic_emittance_measurement,
)
from lcls_tools.common.measurements.emittance_measurement import (
    EmittanceMeasurementResult,
)


class TestGPSRLCLSTools:
    def setup_method(self):
        # Setup code to run before each test method
        self.test_filename = os.path.join(
            test_file_dir, "test_automatic_emittance_scan.h5"
        )
        self.hdf5_file = h5py.File(self.test_filename, "r")

    def test_get_lcls_tools_data_from_file(self):
        # Test the get_lcls_tools_data_from_file function
        data = get_lcls_tools_data_from_file(self.test_filename)

        required_keys = [
            "quad_strengths",
            "quad_pv_values",
            "rmat",
            "resolution",
            "images",
            "design_twiss",
            "energy",
        ]

        for key in required_keys:
            assert key in data

        # check the shapes of the data
        assert data["quad_strengths"].shape == (17,)
        assert data["quad_pv_values"].shape == (17,)
        assert data["rmat"].shape == (17, 6, 6)
        assert data["images"].shape == (17, 104, 139)
        assert isinstance(data["resolution"], float)
        assert np.array(data["design_twiss"]).shape == (4,)
        assert isinstance(data["energy"], float)

    def test_process_automatic_emittance_measurement(self):
        # Test the process_automatic_emittance_measurement function

        # create emittance measurement result object
        obj = hdf5_group_to_dict(self.hdf5_file)
        obj.pop("environment_variables")

        for key in [
            "quadrupole_focusing_strengths",
            "quadrupole_pv_values",
            "bmag",
            "twiss_at_screen",
            "rms_beamsizes",
        ]:
            obj[key] = [values for _, values in obj[key].items()]

        result = EmittanceMeasurementResult(**obj)

        output = process_automatic_emittance_measurement(
            result, n_stds=3, max_pixels=1e5, median_filter_size=3
        )

        n_quad_strengths = output["quad_strengths"].shape[0]

        assert len(output["images"]) == n_quad_strengths
        assert output["images"].shape[0] == n_quad_strengths
        assert output["quad_pv_values"].shape[0] == n_quad_strengths
        assert output["rmat"].shape == (n_quad_strengths, 6, 6)
        assert isinstance(output["resolution"], float)
        assert len(output["design_twiss"]) == 4
        assert output["energy"] == result.metadata["energy"]

    # cleanup
    def teardown_method(self):
        self.hdf5_file.close()
