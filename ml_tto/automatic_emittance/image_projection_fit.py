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

from typing import Callable, Literal
import scipy
from skimage.measure import block_reduce
from skimage.filters import threshold_triangle
from scipy.ndimage import median_filter, gaussian_filter
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt


def compute_blob_stats(image):
    """
    Compute the center (centroid) and RMS size of a blob in a 2D image
    using intensity-weighted averages.

    Parameters:
        image (np.ndarray): 2D array representing the image.

    Returns:
        center (tuple): (x_center, y_center)
        rms_size (tuple): (x_rms, y_rms)
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array")

    # Get coordinate grids
    y_indices, x_indices = np.indices(image.shape)

    # Flatten everything
    x = x_indices.ravel()
    y = y_indices.ravel()
    weights = image.ravel()

    # Total intensity
    total_weight = np.sum(weights)
    if total_weight == 0:
        raise ValueError(
            "Total image intensity is zero — can't compute centroid or RMS size."
        )

    # Weighted centroid
    x_center = np.sum(x * weights) / total_weight
    y_center = np.sum(y * weights) / total_weight

    # Weighted RMS size
    x_rms = np.sqrt(np.sum(weights * (x - x_center) ** 2) / total_weight)
    y_rms = np.sqrt(np.sum(weights * (y - y_center) ** 2) / total_weight)

    return np.array((x_rms, y_rms)), np.array((x_center, y_center))


def process_images(
    images: np.ndarray,
    pixel_size: float,
    image_fitter: Callable = compute_blob_stats,
    pool_size: Optional[int] = None,
    median_filter_size: Optional[int] = None,
    threshold: Optional[float] = None,
    threshold_multiplier: float = 1.0,
    n_stds: int = 8,
    center_images: bool = False,
    visualize: bool = False,
    return_raw_cropped: bool = False,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Process a batch of images for use in GPSR.
    The images are cropped, thresholded, pooled, median filtered, and normalized.
    An image_fitter function is used to fit the images and return the rms size and centroid to crop the images.

    Optionally, the images can be centered using the image_fitter function.

    Parameters
    ----------
    images : np.ndarray
        A batch of images with shape (..., H, W).
    pixel_size : float
        Pixel size of the screen in microns.
    image_fitter : Callable
        A function that fits an image and returns the rms size and centroid as a tuple in px coordinates.
        Example: <rms size>, (<x_center>, <y_center>) = image_fitter(image)
    threshold : float, optional
        The threshold to apply to the images before pooling and filters, by default None. If None, the threshold is calculated via the triangle method.
    threshold_multiplier : float, optional
        The multiplier for the threshold, by default 1.0.
    pool_size : int, optional
        The size of the pooling window, by default None. If None, no pooling is applied.
    median_filter_size : int, optional
        The size of the median filter, by default None. If None, no median filter is applied.
    n_stds : int, optional
        The number of standard deviations to crop the images, by default 8.
    normalization : str, optional
        Normalization method: 'independent' (default) or 'max_intensity_image'.
    center_images : bool, optional
        Whether to center the images before processing, by default False.
        If True, the images are centered using the image_fitter function.
    visualize : bool, optional
        Whether to visualize the images at each step of the processing, by default False.
        If True, the images are displayed using matplotlib.
    return_raw_cropped: bool, optional
        If true, return the raw image cropped.

    Returns
    -------
    np.ndarray
        The processed images with cropped shape (..., H', W').
    np.ndarray
        The meshgrid for the processed images.

    """

    batch_shape = images.shape[:-2]
    batch_dims = tuple(range(len(batch_shape)))
    center_location = np.array(images.shape[-2:]) // 2
    center_location = center_location[::-1]

    raw_images = np.copy(images)

    if visualize:
        plt.figure()
        plt.imshow(images[(0,) * len(batch_shape)])

    # median filter
    if median_filter_size is not None:
        images = median_filter(
            images,
            size=median_filter_size,
            axes=[-2, -1],
        )

    # apply threshold if provided -- otherwise calculate threshold using triangle method
    if threshold is None:
        avg_image = np.mean(images, axis=batch_dims)
        threshold = threshold_triangle(avg_image)
    images = np.clip(images - threshold_multiplier * threshold, 0, None)

    # median filter -- 2nd application
    if median_filter_size is not None:
        images = median_filter(
            images,
            size=median_filter_size,
            axes=[-2, -1],
        )

    if visualize:
        plt.figure()
        plt.title("post filtering and thresholding")
        plt.imshow(images[(0,) * len(batch_shape)])

    # center the images
    if center_images:
        # flatten batch dimensions
        images = np.reshape(images, (-1,) + images.shape[-2:])
        centered_images = np.zeros_like(images)

        for i in range(images.shape[0]):
            # fit the image centers
            rms_size, centroid = image_fitter(images[i])

            # shift the images to center them
            centered_images[i] = scipy.ndimage.shift(
                images[i],
                -(centroid - center_location)[::-1],
                order=1,  # linear interpolation to avoid artifacts
            )

            if visualize:
                fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
                ax[0].imshow(images[i])
                ax[0].plot(*centroid, "r+")
                ax[1].imshow(centered_images[i])
                ax[1].plot(*center_location, "r+")

        # reshape back to original shape
        images = np.reshape(centered_images, batch_shape + images.shape[-2:])

    if visualize:
        plt.figure()
        plt.title("post image centering")
        plt.imshow(images[(0,) * len(batch_shape)])

    total_image = np.mean(images, axis=batch_dims)
    rms_size, centroid = image_fitter(total_image)
    centroid = centroid[::-1]
    rms_size = rms_size[::-1]

    crop_ranges = np.array(
        [
            (centroid - n_stds * rms_size).astype("int"),
            (centroid + n_stds * rms_size).astype("int"),
        ]
    )

    # transpose crop ranges temporarily to clip on image size
    crop_ranges = crop_ranges.T
    crop_ranges[0] = np.clip(crop_ranges[0], 0, images.shape[-2])
    crop_ranges[1] = np.clip(crop_ranges[1], 0, images.shape[-1])

    if visualize:
        plt.figure()
        plt.imshow(total_image)
        plt.plot(*centroid[::-1], "+r")
        rect = plt.Rectangle(
            (crop_ranges[1][0], crop_ranges[0][0]),
            crop_ranges[1][1] - crop_ranges[1][0],
            crop_ranges[0][1] - crop_ranges[0][0],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        plt.gca().add_patch(rect)

    cropped_widths = [
        crop_ranges[1][1] - crop_ranges[1][0],
        crop_ranges[0][1] - crop_ranges[0][0],
    ]
    centroid = [
        (crop_ranges[1][1] + crop_ranges[1][0]) / 2,
        (crop_ranges[0][1] + crop_ranges[0][0]) / 2,
    ]
    if return_raw_cropped:
        images = raw_images[
            ...,
            crop_ranges[0][0] : crop_ranges[0][1],
            crop_ranges[1][0] : crop_ranges[1][1],
        ]
    else:
        images = images[
            ...,
            crop_ranges[0][0] : crop_ranges[0][1],
            crop_ranges[1][0] : crop_ranges[1][1],
        ]

    if visualize:
        plt.figure()
        plt.title("post cropping")
        plt.imshow(images[(0,) * len(batch_shape)])

    # pooling
    if pool_size is not None:
        block_size = (1,) * len(batch_shape) + (pool_size,) * 2
        images = block_reduce(images, block_size=block_size, func=np.mean)

    # compute meshgrids for screens
    bins = []
    pool_size = 1 if pool_size is None else pool_size

    # returns left sided bins
    for j in [-2, -1]:
        img_bins = np.arange(images.shape[j])
        img_bins = img_bins - len(img_bins) / 2
        img_bins = img_bins * pixel_size * 1e-6 * pool_size
        bins += [img_bins]

    return images, centroid, cropped_widths


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
        4.0, description="Number of standard deviations to use for the bounding box"
    )
    projection_fit: Optional[ProjectionFit] = MLProjectionFit(
        model=MLGaussianModel(use_priors=True), relative_filter_size=0.01
    )
    initial_filter_size: PositiveInt = 5
    visualize: bool = False

    def _fit_image(self, image: np.ndarray) -> ImageProjectionFitResult:
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

        if self.visualize:
            plot_image_projection_fit(result)

        return result
