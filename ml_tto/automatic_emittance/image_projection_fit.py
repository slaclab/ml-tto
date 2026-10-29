import numpy as np
from pydantic import (
    PositiveFloat,
    Field,
    PositiveInt,
)

from slac_measurements.image.fit import ImageProjectionFit
from slac_measurements.image.processing import process_images


class RecursiveImageProjectionFit(ImageProjectionFit):
    n_stds: PositiveFloat = Field(
        4.0, description="Number of standard deviations to use for the bounding box"
    )
    initial_filter_size: PositiveInt = 3
    visualize: bool = False

    def _fit_image(self, image: np.ndarray):
        """
        Fit the image recursively by cropping the image to the bounding box of the first fit
        and then refitting the image. This is done to avoid fitting the background noise
        and to get a more accurate fit of the beam size and location.
        Images and crop offsets follow slac-measurements' (x, y) axis order.
        """
        cropped_image, offsets = process_images(
            image,
            n_stds=self.n_stds,
            median_filter_size=self.initial_filter_size,
            crop=True,
            center=False,
        )
        result = super()._fit_image(cropped_image)

        for i in range(2):
            if np.isfinite(result.rms_size[i]):
                offset = offsets.reshape(-1, 2)[0, i]
                result.centroid[i] += offset
                result.beam_extent[i] += offset

        return result
