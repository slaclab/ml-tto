from copy import copy, deepcopy
from typing import List, Optional
import warnings

import numpy as np
from numpy import ndarray
from pydantic import ConfigDict, PositiveFloat, Field, PositiveInt, confloat
import scipy
import scipy.ndimage
import scipy.signal
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
    beam_extent: NDArrayAnnotatedType = Field(
        description="Extent of the beam in the data, defined as mean +/- 2*sigma"
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
        "sigma": Parameter(bounds=[1e-8, 5.0]),
        "amplitude": Parameter(bounds=[0.01, 1.0]),
        "offset": Parameter(bounds=[0.01, 1.0]),
    },
)


class MLGaussianModel(GaussianModel):
    parameters: ModelParameters = ml_gaussian_parameters

    def find_init_values(self) -> dict:
        """Fit data without optimization, return values."""

        data = self._profile_data
        x = np.linspace(0, 1, len(data))
        offset = data.min() + 0.01
        amplitude = data.max() - offset

        truncated_data = np.clip(data - 2.0 * offset, 0.01 * offset, None)
        weighted_mean = np.average(x, weights=truncated_data)
        weighted_sigma = np.sqrt(np.cov(x, aweights=truncated_data))

        init_values = {
            "mean": weighted_mean,
            "sigma": weighted_sigma,
            "amplitude": amplitude,
            "offset": offset,
        }
        self.parameters.initial_values = init_values
        return init_values

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
        # sigma_mean = init_values["sigma"]
        # sigma_var = 0.5 * sigma_mean
        # sigma_alpha = (sigma_mean**2) / sigma_var
        # sigma_beta = sigma_mean / sigma_var
        # sigma_prior = gamma(sigma_alpha, loc=0, scale=1 / sigma_beta)
        sigma_prior = uniform(1e-8, 5.0)

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
    beam_extent_n_stds: PositiveFloat = Field(
        2.0,
        description="Number of standard deviations on either side to use for the beam extent",
    )
    # max_sigma_to_image_size_ratio: PositiveFloat = Field(
    #    2.0, description="Maximum sigma to projection size ratio"
    # )

    def _fit_image(self, image: ndarray) -> ImageProjectionFitResult:
        fit_parameters = []
        noise_stds = []
        signal_to_noise_ratios = []
        beam_extent = []

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

            # calculate the extent of the beam in the projection - scaled to the image size
            beam_extent.append(
                [
                    parameters["mean"] - self.beam_extent_n_stds * parameters["sigma"],
                    parameters["mean"] + self.beam_extent_n_stds * parameters["sigma"],
                ]
            )

            # if the beam extent is outside the image then its off the screen etc. and fits cannot be trusted
            if beam_extent[-1][0] < 0 or beam_extent[-1][1] > len(projection):
                for name in parameters.keys():
                    parameters[name] = np.nan

                warnings.warn(
                    f"Projection in {direction[i]} was off the screen, fit cannot be trusted"
                )

                continue

        result = ImageProjectionFitResult(
            centroid=[ele["mean"] for ele in fit_parameters],
            rms_size=[ele["sigma"] for ele in fit_parameters],
            total_intensity=image.sum(),
            projection_fit_parameters=fit_parameters,
            image=image,
            projection_fit_method=self.projection_fit.model,
            noise_std=noise_stds,
            signal_to_noise_ratio=signal_to_noise_ratios,
            beam_extent=beam_extent,
        )

        return result


class RecursiveImageProjectionFit(ImageProjectionFit):
    n_stds: PositiveFloat = Field(
        8.0, description="Number of standard deviations to use for the bounding box"
    )
    projection_fit: Optional[ProjectionFit] = MLProjectionFit(
        model=MLGaussianModel(use_priors=True), relative_filter_size=0.01
    )
    initial_filter_size: PositiveInt = 10
    visualize: bool = False

    def _fit_image(self, image: np.ndarray) -> ImageProjectionFitResult:
        """
        Fit the image recusrively by cropping the image to the bounding box of the first fit
        and then refitting the image. This is done to avoid fitting the background noise
        and to get a more accurate fit of the beam size and location.

        The fit is done in several steps:
        1. Fit the image to get the initial beam size and location
        2. If the fit is successful in either direction
            a. crop the image to the bounding box of the fit
            b. refit the image to get a more accurate beam size and location
            c. update the fit parameters to reflect the new image size
            d. recalculate the noise std and signal to noise ratio
        3. If the fit is not successful in either direction then return the original image and fit parameters

        """
        initial_fit = ImageProjectionFit(signal_to_noise_threshold=0.01)
        fresult = initial_fit.fit_image(scipy.ndimage.gaussian_filter(image, self.initial_filter_size))

        if self.visualize:
            plot_image_projection_fit(fresult)

        rms_size = np.array(fresult.rms_size)
        centroid = np.array(fresult.centroid)

        # if all rms sizes are nan then we can't crop the image
        if np.all(np.isnan(rms_size)):
            return fresult

        # get ranges for clipping
        crop_ranges = []
        for i in range(2):
            # if the rms size is nan then we can't crop this direction
            if np.isnan(rms_size[i]):
                r = np.array([0, image.shape[i]]).astype(int)
            else:
                # set a minimum size for the crop to avoid cropping too small
                half_width = np.max((10.0, rms_size[i] * self.n_stds))
                r = np.array(
                    [
                        centroid[i] - half_width,
                        centroid[i] + half_width,
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

        # do final fit
        self.beam_extent_n_stds = 2.0
        result = super()._fit_image(cropped_image)

        # if the fit along an axis is successful then update the fit parameters
        for i in range(2):
            if np.isfinite(result.rms_size[i]) and np.isfinite(centroid[i]):
                # we cropped in this direction so we need to update the fit parameters
                result.centroid[i] += centroid[i] - crop_widths[i] / 2 + 0.5
                result.beam_extent[i] += centroid[i] - crop_widths[i] / 2 + 0.5

        if self.visualize:
            plot_image_projection_fit(result)

        return result
