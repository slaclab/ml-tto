import os

import numpy as np

from lcls_tools.common.measurements.screen_profile import (
    ScreenBeamProfileMeasurementResult,
)
from lcls_tools.common.image.processing import ImageProcessor
from ml_tto.automatic_emittance.image_projection_fit import ImageProjectionFit
from ml_tto.saver import H5Saver


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
            "i": np.array((1.0, 2.0), dtype="O"),
        }
        saver.save_to_h5(data, "test.h5")
        os.remove("test.h5")

    def test_screen_measurement_results(self):
        # Load test data
        images = np.load("ml_tto/tests/fixtures/test_images.npy")

        # Process data
        image_processor = ImageProcessor()
        beam_fit = ImageProjectionFit()

        processed_images = [image_processor.auto_process(image) for image in images]

        rms_sizes = []
        centroids = []
        total_intensities = []
        for image in processed_images:
            fit_result = beam_fit.fit_image(image)
            rms_sizes.append(fit_result.rms_size)
            centroids.append(fit_result.centroid)
            total_intensities.append(fit_result.total_intensity)

        # Store results in ScreenBeamProfileMeasurementResult
        result = ScreenBeamProfileMeasurementResult(
            raw_images=images,
            processed_images=processed_images,
            rms_sizes=rms_sizes or None,
            centroids=centroids or None,
            total_intensities=total_intensities or None,
            metadata={"info": "test"},
        )

        # Dump to H5
        result_dict = result.model_dump()
        saver = H5Saver()
        saver.save_to_h5(
            result_dict,
            os.path.join("screen_test.h5"),
        )

        # Load H5
        loaded_dict = saver.load_from_h5("screen_test.h5")

        # Check if the loaded dictionary is the same as the original
        assert result_dict.keys() == loaded_dict.keys()
        assert result_dict["metadata"] == loaded_dict["metadata"]
        assert isinstance(loaded_dict["raw_images"], np.ndarray)
        assert np.allclose(images, loaded_dict["raw_images"], rtol=1e-5)

        mask = ~np.isnan(rms_sizes)
        assert np.allclose(
            np.asarray(rms_sizes)[mask], loaded_dict["rms_sizes"][mask], rtol=1e-5
        )
        mask = ~np.isnan(centroids)
        assert np.allclose(
            np.asarray(centroids)[mask], loaded_dict["centroids"][mask], rtol=1e-5
        )
        assert np.allclose(
            total_intensities, loaded_dict["total_intensities"], rtol=1e-5
        )

        os.remove("screen_test.h5")
