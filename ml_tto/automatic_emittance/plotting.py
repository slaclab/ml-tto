from lcls_tools.common.frontend.plotting.image import (
    plot_image_projection_fit,
)


def plot_screen_profile_measurement(measurement):
    """
    Plot the screen profile measurement result.

    Parameters:
        measurement (ScreenBeamProfileMeasurement): The screen beam profile measurement object.
        n_stds (float): Number of standard deviations for bounding box.

    Returns:
        fig, ax: The figure and axes objects.
    """

    result = measurement.beam_fit.fit_image(
        measurement.image_processor.process(measurement.beam_profile_device.image)
    )

    return plot_image_projection_fit(result)
