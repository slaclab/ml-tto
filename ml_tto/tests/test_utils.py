from lcls_tools.common.image.roi import CircularROI, ROI
from unittest.mock import MagicMock
import numpy as np

from lcls_tools.common.measurements.screen_profile import (
    ScreenBeamProfileMeasurementResult,
)
from ml_tto.automatic_emittance.utils import (
    calculate_bounding_box_coordinates,
    calculate_bounding_box_penalty,
    validate_beamsize_measurement_result,
)


class TestUtils:
    def test_validate_beamsize_measurement_result(self):
        # Mock ScreenBeamProfileMeasurementResult
        mock_result = MagicMock(ScreenBeamProfileMeasurementResult)
        mock_result.total_intensities = np.array([1e4, 1e5, 1e2])
        rms_sizes = np.stack([np.array([1, 1]), np.array([2, 2]), np.array([3, 3])])
        mock_result.rms_sizes = rms_sizes
        mock_result.centroids = (
            np.stack([np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]) + 1
        )
        mock_result.metadata = MagicMock()
        mock_result.metadata.image_processor.roi = CircularROI(
            center=np.array([1, 1]), radius=2
        )

        validated_result, bb_penalties, log10_total_intensity = (
            validate_beamsize_measurement_result(
                mock_result,
                roi=mock_result.metadata.image_processor.roi,
                min_log10_intensity=3.0, n_stds=2.0
            )
        )

        # Assertions
        # note that the last bb_penalty is NaN because the total intensity is below the threshold
        assert np.allclose(
            bb_penalties[:2],
            np.array([np.linalg.norm(ele) - 2.0 for ele in rms_sizes[:2]]),
        )
        assert np.isnan(bb_penalties[2])

        assert np.allclose(
            log10_total_intensity, np.log10(mock_result.total_intensities)
        )
        assert np.allclose(validated_result.rms_sizes[0], np.array([1, 1]))
        assert np.isnan(validated_result.rms_sizes[1]).all()
        assert np.isnan(validated_result.rms_sizes[2]).all()
        assert np.allclose(validated_result.centroids[0], np.array([1, 1]))
        assert np.isnan(validated_result.centroids[1]).all()
        assert np.isnan(validated_result.centroids[2]).all()

    def test_calculate_bounding_box_coordinates(self):
        # Mock ImageFitResult
        rms_size = np.array([2, 4]).reshape(1, 2)
        centroid = np.array([1, 1]).reshape(1, 2)

        expected_bbox_coords = [
            np.array([0, -1]),
            np.array([2, 3]),
            np.array([0, 3]),
            np.array([2, -1]),
        ]

        bbox_coords = calculate_bounding_box_coordinates(rms_size, centroid, n_stds=1)

        # Assertions
        for coord, expected_coord in zip(bbox_coords, expected_bbox_coords):
            assert np.allclose(coord, expected_coord)

    def test_calculate_bounding_box_penalty(self):
        rms_size = np.array([2, 4]).reshape(1, 2)
        centroid = np.array([1, 1]).reshape(1, 2)
        bbox_coords = calculate_bounding_box_coordinates(rms_size, centroid, n_stds=2)

        roi = CircularROI(center=[1, 1], radius=1)
        penalty = calculate_bounding_box_penalty(roi, bbox_coords)
        assert penalty == np.linalg.norm(roi.center - np.array((3.0, 5.0))) - 1.0
