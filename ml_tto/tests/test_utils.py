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
        centroids = np.stack([np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]) + 2
        mock_result.centroids = centroids
        mock_result.metadata = MagicMock()
        radius = 2
        mock_result.metadata.image_processor.roi = CircularROI(
            center=np.array([1, 1]), radius=radius
        )
        n_stds = 1.0
        validated_result, bb_penalties, log10_total_intensity = (
            validate_beamsize_measurement_result(
                mock_result,
                roi=mock_result.metadata.image_processor.roi,
                min_log10_intensity=3.0, n_stds=n_stds
            )
        )

        # Assertions
        # note that the last bb_penalty is NaN because the total intensity is below the threshold
        assert np.allclose(
            bb_penalties[:2],
            np.array([np.linalg.norm(
                np.ones(2)*radius - (n_stds*size + centroid)
                ) - radius for centroid, size in zip(centroids[:2], rms_sizes[:2])])
        )
        assert np.isnan(bb_penalties[2])

        assert np.allclose(
            log10_total_intensity, np.log10(mock_result.total_intensities)
        )
        assert np.allclose(validated_result.rms_sizes[0], np.array([1, 1]))
        assert np.isnan(validated_result.rms_sizes[1]).all()
        assert np.isnan(validated_result.rms_sizes[2]).all()
        assert np.allclose(validated_result.centroids[0], np.array([2, 2]))
        assert np.isnan(validated_result.centroids[1]).all()
        assert np.isnan(validated_result.centroids[2]).all()

    def test_calculate_bounding_box_coordinates(self):
        # Mock ImageFitResult
        rms_size = np.array([2, 4]).reshape(1, 2)
        centroid = np.array([1, 1]).reshape(1, 2)

        expected_bbox_coords = [
            np.array([-1, -3]),
            np.array([3, 5]),
            np.array([-1, 5]),
            np.array([3, -3]),
        ]

        bbox_coords = calculate_bounding_box_coordinates(rms_size, centroid, n_stds=1)
        # Assertions
        for coord, expected_coord in zip(bbox_coords, expected_bbox_coords):
            assert np.allclose(coord, expected_coord)

    def test_calculate_bounding_box_penalty(self):
        rms_size = np.array([2, 4]).reshape(1, 2)
        centroid = np.array([1, 1]).reshape(1, 2)
        bbox_coords = calculate_bounding_box_coordinates(rms_size, centroid, n_stds=1)

        roi = CircularROI(center=[1, 1], radius=1)
        penalty = calculate_bounding_box_penalty(roi, bbox_coords)
        assert penalty == np.linalg.norm(roi.center - np.array((3.0, 5.0))) - 1.0
