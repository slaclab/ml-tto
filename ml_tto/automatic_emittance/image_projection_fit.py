from copy import copy, deepcopy
from typing import List, Optional
import warnings

import numpy as np
from numpy import ndarray
from pydantic import ConfigDict, PositiveFloat, Field, confloat
import scipy
from scipy.stats import norm, gamma, uniform

from lcls_tools.common.data.fit.methods import GaussianModel
from lcls_tools.common.data.fit.projection import ProjectionFit
from lcls_tools.common.image.fit import ImageProjectionFit, ImageFitResult
from lcls_tools.common.data.fit.method_base import MethodBase
from lcls_tools.common.measurements.utils import NDArrayAnnotatedType


from lcls_tools.common.data.fit.method_base import (
    ModelParameters,
    Parameter,
)

from ml_tto.automatic_emittance.plotting import plot_image_projection_fit


class ImageProjectionFitResult(ImageFitResult):
    projection_fit_method: MethodBase
    projection_fit_parameters: List[dict[str, float]]
    noise_std: NDArrayAnnotatedType = Field(
        description="Standard deviation of the noise in the data"
    )
    signal_to_noise_ratio: NDArrayAnnotatedType = Field(
        description="Ratio of fit amplitude to noise std in the data"
    )


class MLProjectionFit(ProjectionFit):
    """
    1d fitting class that allows users to choose the model with which the fit
    is performed, and if prior assumptions (bayesian regression) about
    the data should be used when performing the fit.
    Additionally there is an option to visualize the fitted data and priors.
    -To perform a 1d fit, call fit_projection(projection_data={*data_to_fit*})
    ------------------------
    Arguments:
    model: MethodBase (this argument is a child class object of method base
        e.g GaussianModel & DoubleGaussianModel)
    visualize_priors: bool (shows plots of the priors and init guess
                      distribution before fit)
    use_priors: bool (incorporates prior distribution information into fit)
    visualize_fit: bool (visualize the parameters as a function of the
                   forward function
        from our model compared to distribution data)
    """

    relative_filter_size: confloat(ge=0, le=1) = 0.0

    def model_setup(self, projection_data=np.ndarray) -> None:
        """sets up the model and init_values/priors"""
        # apply a gaussian filter to the data to smooth
        filter_size = int(len(projection_data) * self.relative_filter_size)

        if filter_size > 0:
            projection_data = scipy.ndimage.gaussian_filter1d(
                projection_data, filter_size
            )

        self.model.profile_data = projection_data


ml_gaussian_parameters = ModelParameters(
    name="Gaussian Parameters",
    parameters={
        "mean": Parameter(bounds=[0.01, 1.0]),
        "sigma": Parameter(bounds=[0.00001, 5.0]),
        "amplitude": Parameter(bounds=[0.01, 1.0]),
        "offset": Parameter(bounds=[0.01, 1.0]),
    },
)


class MLGaussianModel(GaussianModel):
    parameters: ModelParameters = ml_gaussian_parameters

    def find_priors(self, **kwargs) -> dict:
        """
        Do initial guesses based on data and make distribution from that guess.
        """
        init_values = self.find_init_values()
        amplitude_mean = init_values["amplitude"]
        amplitude_var = 0.1
        amplitude_alpha = (amplitude_mean**2) / amplitude_var
        amplitude_beta = amplitude_mean / amplitude_var
        amplitude_prior = gamma(amplitude_alpha, loc=0, scale=1 / amplitude_beta)

        # Creating a normal distribution of points around the inital mean.
        mean_prior = norm(init_values["mean"], 0.1)
        # mean_prior = uniform(0.0001, 1.0)
        # sigma_alpha = 2.5
        # sigma_beta = 5.0
        # sigma_prior = gamma(sigma_alpha, loc=0, scale=1 / sigma_beta)
        sigma_prior = uniform(0.00001, 5.0)

        # Creating a normal distribution of points around initial offset.
        offset_prior = norm(init_values["offset"], 0.5)
        priors = {
            "mean": mean_prior,
            "sigma": sigma_prior,
            "amplitude": amplitude_prior,
            "offset": offset_prior,
        }

        self.parameters.priors = priors
        return priors


class ImageProjectionFit(ImageProjectionFit):
    """
    Image fitting class that gets the beam size and location by independently fitting
    the x/y projections. The default configuration uses a Gaussian fitting of the
    profile with prior distributions placed on the model parameters.
    """

    projection_fit: Optional[ProjectionFit] = MLProjectionFit(
        model=MLGaussianModel(use_priors=True), relative_filter_size=0.01
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    signal_to_noise_threshold: PositiveFloat = Field(
        4.0, description="Fit amplitude to noise threshold for the fit"
    )
    max_sigma_to_image_size_ratio: PositiveFloat = Field(
        2.0, description="Maximum sigma to projection size ratio"
    )

    def _fit_image(self, image: ndarray) -> ImageProjectionFitResult:
        fit_parameters = []
        noise_stds = []
        signal_to_noise_ratios = []

        direction = ["x", "y"]
        for i in range(2):
            projection = np.array(np.sum(image, axis=i))
            parameters = self.projection_fit.fit_projection(projection)

            # determine the noise around the projection fit
            x = np.arange(len(projection))
            noise_std = np.std(
                self.projection_fit.model.forward(x, parameters) - projection
            )
            noise_stds.append(noise_std)
            signal_to_noise_ratios.append(parameters["amplitude"] / noise_std)

            # if the amplitude of the the fit is smaller than noise then reject
            if signal_to_noise_ratios[-1] < self.signal_to_noise_threshold:
                for name in parameters.keys():
                    parameters[name] = np.nan

                warnings.warn(
                    f"Projection in {direction[i]} had a low amplitude relative to noise"
                )

            fit_parameters.append(parameters)

            # if 4*sigma does not fit on the projection then its too big
            # if self.max_sigma_to_image_size_ratio * params["sigma"] > len(
            #    projections[i]
            # ):
            #    for name in params.keys():
            #        params[name] = np.nan

            #    warnings.warn(
            #        f"Projection in {direction[i]} was too big relative to projection span"
            #    )

            #    continue

        result = ImageProjectionFitResult(
            centroid=[ele["mean"] for ele in fit_parameters],
            rms_size=[ele["sigma"] for ele in fit_parameters],
            total_intensity=image.sum(),
            projection_fit_parameters=fit_parameters,
            image=image,
            projection_fit_method=self.projection_fit.model,
            noise_std=noise_stds,
            signal_to_noise_ratio=signal_to_noise_ratios,
        )

        return result


class RecursiveImageProjectionFit(ImageProjectionFit):
    n_stds: PositiveFloat = Field(
        4.0, description="Number of standard deviations to use for the bounding box"
    )
    show_intermediate_plots: bool = Field(
        False, description="Show intermediate plots of the cropped image and fit"
    )

    def _fit_image(self, image: np.ndarray) -> ImageProjectionFitResult:
        fresult = super()._fit_image(image)

        rms_size = np.array(fresult.rms_size)
        centroid = np.array(fresult.centroid)

        # if all rms sizes are nan then we can't crop the image
        if np.all(np.isnan(rms_size)):
            return fresult

        # get ranges for clipping
        crop_ranges = []
        for i in range(2):
            if np.isnan(rms_size[i]):
                # if the rms size is nan then we can't crop this direction
                r = np.array([0, image.shape[i]]).astype(int)
            else:
                r = np.array(
                    [
                        centroid[i] - rms_size[i] * self.n_stds,
                        centroid[i] + rms_size[i] * self.n_stds,
                    ]
                ).astype(int)
                r = np.clip(r, 0, image.shape[i])

            crop_ranges.append(r)

        crop_ranges = np.array(crop_ranges)
        crop_widths = crop_ranges[:, 1] - crop_ranges[:, 0] + 1

        # crop the image based on the bounding box
        cropped_image = image[
            crop_ranges[1][0] : crop_ranges[1][1], crop_ranges[0][0] : crop_ranges[0][1]
        ]
        result = super()._fit_image(cropped_image)
        if self.show_intermediate_plots:
            plot_image_projection_fit(result)

        # update the image
        # result.image = image

        # if the fit along an axis is successful then update the fit parameters
        for i in range(2):
            if not np.isnan(result.rms_size[i]):
                # we cropped in this direction so we need to update the fit parameters
                # note that we cropped when the centroid/rms is not nan
                if not np.isnan(centroid[i]):
                    result.centroid[i] += centroid[i] - crop_widths[i] / 2
                    # result.projection_fit_parameters[i]["mean"] = result.centroid[i]

                # recalculate the noise std and signal to noise ratio
                projection = np.sum(result.image, axis=i)
                x = np.arange(len(projection))
                noise_std = np.std(
                    self.projection_fit.model.forward(
                        x, result.projection_fit_parameters[i]
                    )
                    - projection
                )
                result.noise_std[i] = noise_std
                result.signal_to_noise_ratio[i] = (
                    result.projection_fit_parameters[i]["amplitude"] / noise_std
                )

            else:
                # if the fit is not successful then revert the parameters back to the original
                # parameters
                result.centroid[i] = fresult.centroid[i]
                result.rms_size[i] = fresult.rms_size[i]
                result.projection_fit_parameters[i] = fresult.projection_fit_parameters[
                    i
                ]
                result.noise_std[i] = fresult.noise_std[i]
                result.signal_to_noise_ratio[i] = fresult.signal_to_noise_ratio[i]

        return result
