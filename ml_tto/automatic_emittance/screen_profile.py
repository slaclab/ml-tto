from typing import Any
import numpy as np
import logging

from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, retry_if_exception_type, Retrying, RetryError

from lcls_tools.common.devices.screen import Screen
from ml_tto.automatic_emittance.image_projection_fit import ImageProjectionFit
from ml_tto.errors import NoBeamError
from lcls_tools.common.image.fit import ImageFit
from lcls_tools.common.image.processing import ImageProcessor
from lcls_tools.common.measurements.screen_profile import ScreenBeamProfileMeasurement
from pydantic import (
    ConfigDict,
    SerializeAsAny,
)
from typing import Optional

from lcls_tools.common.measurements.utils import NDArrayAnnotatedType
import lcls_tools

logger = logging.getLogger("screen_profile_measurement")

class ScreenBeamProfileMeasurementResult(lcls_tools.common.BaseModel):
    """
    Class that contains the results of a beam profile measurement

    Attributes
    ----------
    raw_images : ndarray
        Numpy array of raw images taken during the measurement
    processed_images : ndarray
        Numpy array of processed images taken during the measurement
    rms_sizes : ndarray
        Numpy array of rms sizes of the beam in pixel units.
    centroids : ndarray
        Numpy array of centroids of the beam in pixel units.
    total_intensities : ndarray
        Numpy array of total intensities of the beam.
    metadata : Any
        Metadata information related to the measurement.

    """

    raw_images: NDArrayAnnotatedType
    processed_images: NDArrayAnnotatedType
    rms_sizes: Optional[NDArrayAnnotatedType] = None
    centroids: Optional[NDArrayAnnotatedType] = None
    total_intensities: Optional[NDArrayAnnotatedType] = None
    signal_to_noise_ratios: Optional[NDArrayAnnotatedType] = None
    metadata: SerializeAsAny[Any]

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class ScreenBeamProfileMeasurement(ScreenBeamProfileMeasurement):
    """
    Class that allows for beam profile measurements and fitting
    ------------------------
    Arguments:
    name: str (name of measurement default is beam_profile),
    device: Screen (device that will be performing the measurement),
    beam_fit: method for performing beam profile fit, default is gfit
    fit_profile: bool = True
    ------------------------
    Methods:
    single_measure: measures device and returns raw and processed image
    measure: does multiple measurements and has an option to fit the image
             profiles

    #TODO: DumpController?
    #TODO: return images flag
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "beam_profile"
    image_processor: Optional[ImageProcessor] = ImageProcessor()
    beam_fit: ImageFit = ImageProjectionFit()
    fit_profile: bool = True

    def measure(self, n_shots: int = 1) -> dict:
        """
        Measurement function that takes in n_shots as argument
        where n_shots is the number of image profiles
        we would like to measure. Invokes single_measure per shot,
        storing them in a dictionary sorted by shot number
        Then if self.fit_profile = True, fits the profile of the beam
        and concatenates results with the image dictionary sorted by
        shot number
        """
        images = []
        processed_images = []
        rms_sizes = []
        centroids = []
        total_intensities = []
        signal_to_noise_ratios = []
        while len(images) < n_shots:
            try:
                for attempt in Retrying(
                    stop=stop_after_attempt(3),
                    retry=retry_if_exception_type(NoBeamError)
                ):
                    with attempt:
                        logger.debug("getting image")
                        image = self.beam_profile_device.image
                        # TODO: need to add a wait statement in here for images to update
            
                        processed_image = self.image_processor.auto_process(image) 
                        fit_result = self.beam_fit.fit_image(image)
                        if np.all(np.isnan(fit_result.rms_size)):
                            raise NoBeamError

                        images.append(image)
                        processed_images.append(processed_image)
                        rms_sizes.append(fit_result.rms_size)
                        centroids.append(fit_result.centroid)
                        total_intensities.append(fit_result.total_intensity)
                        signal_to_noise_ratios.append(fit_result.signal_to_noise_ratio)
                        
            except RetryError:
                # append the last fit result which contains nans
                logger.warning("beam projection intensity on screen does not meet signal to noise threshold for either axis")
                images.append(image)                
                processed_images.append(processed_image)
                rms_sizes.append(fit_result.rms_size)
                centroids.append(fit_result.centroid)
                total_intensities.append(fit_result.total_intensity)
                signal_to_noise_ratios.append(fit_result.signal_to_noise_ratio)

        return ScreenBeamProfileMeasurementResult(
            raw_images=images,
            processed_images=processed_images,
            rms_sizes=rms_sizes or None,
            centroids=centroids or None,
            total_intensities=total_intensities or None,
            signal_to_noise_ratios=signal_to_noise_ratios or None,
            metadata=self.model_dump(),
        )
