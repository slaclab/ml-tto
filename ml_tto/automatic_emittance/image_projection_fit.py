import numpy as np
from pydantic import (
    PositiveFloat,
    Field,
    PositiveInt,
)

from lcls_tools.common.image.fit import ImageProjectionFit
from lcls_tools.common.image.processing import process_images


class RecursiveImageProjectionFit(ImageProjectionFit):
    n_stds: PositiveFloat = Field(
        4.0, description="Number of standard deviations to use for the bounding box"
    )
    initial_filter_size: PositiveInt = 3
    visualize: bool = False

    def _fit_image(self, image: np.ndarray):
        """
        Fit the image recusrively by cropping the image to the bounding box of the first fit
        and then refitting the image. This is done to avoid fitting the background noise
        and to get a more accurate fit of the beam size and location.
        """
        cropped_image, centroid, crop_widths = process_images(
            image,
            1,
            n_stds=self.n_stds,
            median_filter_size=self.initial_filter_size,
            return_raw_cropped=True,
        )
        result = super()._fit_image(cropped_image)

        for i in range(2):
            if np.isfinite(result.rms_size[i]) and np.isfinite(centroid[i]):
                # we cropped in this direction so we need to update the fit parameters
                result.centroid[i] += centroid[i] - crop_widths[i] / 2
                result.beam_extent[i] += centroid[i] - crop_widths[i] / 2

        return result
