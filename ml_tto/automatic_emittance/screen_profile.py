import numpy as np
import logging

from tenacity import (
    stop_after_attempt,
    retry_if_exception_type,
    Retrying,
    RetryError,
)

from ml_tto.errors import NoBeamError

from lcls_tools.common.measurements.screen_profile import (
    ScreenBeamProfileMeasurement,
    ScreenBeamProfileMeasurementResult,
)


logger = logging.getLogger("screen_profile_measurement")


class RetryScreenBeamProfileMeasurement(ScreenBeamProfileMeasurement):
    def measure(self) -> ScreenBeamProfileMeasurementResult:
        """
        Modify the base class measurement to retry on NoBeamError
        """
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(3),
                retry=retry_if_exception_type(NoBeamError),
            ):
                with attempt:
                    result = super().measure()

                    # return a non-beam error if all sizes are nans
                    if np.all(np.isnan(result.rms_sizes)):
                        raise NoBeamError

                    return result

        except RetryError:
            logger.warning(
                "Failed to measure screen beam profile after 3 attempts. Returning last measurement"
            )
            return result
