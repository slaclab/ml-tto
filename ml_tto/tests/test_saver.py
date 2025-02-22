from ml_tto.saver import H5Saver
import numpy as np

class TestSaver:
    def test_nans(self):
        saver = H5Saver()
        data = {
            "a": np.nan,
            "b": np.inf,
            "c": -np.inf,
            "d": [np.nan, np.inf, -np.inf],
            "e": [np.nan, np.inf, -np.inf, 1.0],
            "f": [np.nan, np.inf, -np.inf, "a"],
            "g": {"a": np.nan, "b": np.inf, "c": -np.inf},
            "h": "np.Nan",
            "i": np.array((1.0,2.0), dtype="O"),
        }
        saver.save_to_h5(data, "test.h5")