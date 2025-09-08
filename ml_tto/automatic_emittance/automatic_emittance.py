import warnings
import traceback
import time
from typing import Callable, Optional, Tuple

import numpy as np
from pydantic import PositiveFloat, PositiveInt
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.numerical_optimizer import GridOptimizer

from ml_tto.automatic_emittance.emittance import (
    QuadScanEmittance,
    EmittanceMeasurementResult,
)
from ml_tto.automatic_emittance.scan_cropping import crop_scan
from ml_tto.automatic_emittance.transmission import TransmissionMeasurement
from ml_tto.gpsr.lcls_tools import process_automatic_emittance_measurement
from ml_tto.gpsr.quadrupole_scan_fitting import gpsr_fit_quad_scan


class MLQuadScanEmittance(QuadScanEmittance):
    """
    Machine learning-based quadrupole scan emittance measurement.

    This class uses Bayesian optimization to explore the quadrupole strength
    and measure the beam size at different quadrupole settings. It uses the
    Xopt library to perform the optimization.

    Attributes:
        scan_values (list[float]): List of quadrupole strengths used in the scan.
        n_initial_points (PositiveInt): Number of initial points for the optimization.
        n_iterations (PositiveInt): Number of iterations for the optimization.
        max_scan_range (Optional[list[float]]): Maximum scan range for the quadrupole strength.
        X (Optional[Xopt]): Xopt object for Bayesian optimization.
        min_signal_to_noise_ratio (float): Minimum signal-to-noise ratio for valid measurements.
        n_interpolate_points (Optional[PositiveInt]): Number of interpolation measurements made in-between BO-chosen points.
        n_grid_points (PositiveInt): Number of grid points for the numerical optimizer.
        min_beamsize_cutoff (float): Minimum beam size cutoff in microns.
        beamsize_cutoff_max (float): Maximum beam size cutoff as a multiple of the minimum beam size measured.
        beta (float): Exploration parameter for the Bayesian optimization.
        visualize_bo (bool): Whether to visualize the Bayesian optimization process.
        visualize_cropping (bool): Whether to visualize the cropping of the scan.
        verbose (bool): Whether to print verbose output during the measurement.

        evaluate_callback (Optional[callable]): Optional callback function to evaluate additional metrics at each quad strength during the scan.
            Should be in the form of `evaluate_callback(inputs: dict, fit_result: ImageProjectionFitResult) -> dict`.
            Additional results will be added to the `X.data` attribute.

    """

    # basic settings for the scan
    n_initial_points: PositiveInt = 5
    n_iterations: PositiveInt = 5
    max_scan_range: Optional[list[float]] = [-10.0, 10.0]

    # visualization settings
    visualize_bo: bool = False
    visualize_cropping: bool = False
    verbose: bool = False

    # more detailed settings for the scan
    min_signal_to_noise_ratio: float = 4.0
    n_interpolate_points: Optional[PositiveInt] = 3
    n_grid_points: PositiveInt = 100
    min_beamsize_cutoff: float = 100.0  # in microns
    beamsize_cutoff_max: float = 3.0
    beta: float = 10000.0
    evaluate_callback: Optional[Callable] = None
    transmission_measurement: Optional[TransmissionMeasurement] = None
    transmission_measurement_constraint: Optional[float] = 0.9
    max_measurement_retries: int = 10
    bctrl_refresh_rate: float = 0.02

    # data storage
    X: Optional[Xopt] = None
    scan_values: Optional[list[float]] = []

    def _evaluate(self, inputs):
        # set quadrupole strength
        if self.verbose:
            print(f"Setting quadrupole strength to {inputs['k']}")
        self.magnet.bctrl = inputs["k"]

        # start by waiting one refresh cycle for bctrl
        # then wait for bact to match bctrl
        # bctrl refresh rate is less than 10 ms
        time.sleep(self.bctrl_refresh_rate)
        while abs(self.magnet.bctrl - self.magnet.bact) > 0.01:
            time.sleep(self.bctrl_refresh_rate)

        if self.verbose:
            print(f"Quadrupole strength bact is {self.magnet.bact}")

        # try up to `max_measurement_retries` times to make the beamsize measurements
        for attempt in range(self.max_measurement_retries):
            try:
                self.measure_beamsize()
                fit_result = self._info[-1]
                self.scan_values.append(inputs["k"])

                # if transmission measurement is set, measure transmission
                extra_measurements = {}
                if self.transmission_measurement is not None:
                    extra_measurements.update(self.transmission_measurement.measure())

                if self.evaluate_callback is not None:
                    additional_results = self.evaluate_callback(
                        inputs=inputs, fit_result=fit_result
                    )
                    extra_measurements.update(additional_results)

                # add extra measurements to fit result metadata
                fit_result.metadata.update(extra_measurements)

                # replace last element of info with validated result
                fit_result.rms_sizes = np.where(
                    fit_result.signal_to_noise_ratios < self.min_signal_to_noise_ratio,
                    np.nan,
                    fit_result.rms_sizes,
                )
                fit_result.centroids = np.where(
                    fit_result.signal_to_noise_ratios < self.min_signal_to_noise_ratio,
                    np.nan,
                    fit_result.centroids,
                )

                self._info[-1] = fit_result

                # collect results
                rms_x = fit_result.rms_sizes[:, 0]
                rms_y = fit_result.rms_sizes[:, 1]

                results = {
                    "x_rms_px_sq": rms_x**2,
                    "y_rms_px_sq": rms_y**2,
                    "min_signal_to_noise_ratio": np.min(
                        fit_result.signal_to_noise_ratios
                    ),
                }
                results.update(extra_measurements)

                if self.verbose:
                    print(f"Results: {results}")

                return results
            except Exception as e:
                last_exception = e

                print(
                    f"There was an issue making the measurement, retrying, attempt {attempt}"
                )
                print(traceback.format_exc())
                if attempt < self.max_measurement_retries - 1:
                    time.sleep(5.0)

        print("giving up")
        raise last_exception

    def create_xopt_object(self, vocs):
        evaluator = Evaluator(function=self._evaluate)
        generator = UpperConfidenceBoundGenerator(
            vocs=vocs,
            beta=self.beta,
            numerical_optimizer=GridOptimizer(n_grid_points=self.n_grid_points),
            n_interpolate_points=self.n_interpolate_points,
            n_monte_carlo_samples=64,
        )
        self.X = Xopt(vocs=vocs, evaluator=evaluator, generator=generator)

    def update_xopt_object(self, vocs):
        self.X.vocs = vocs
        self.X.generator.vocs = vocs

    def reset(self):
        self.scan_values = []
        self._info = []

    def run_iterations(self, dim_name, n_iterations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for _ in range(n_iterations):
                self.update_xopt_object(self.get_vocs(dim_name))

                if self.visualize_bo:
                    self.X.generator.train_model()
                    self.X.generator.visualize_model(
                        exponentiate=True,
                        show_feasibility=True,
                    )

                self.X.step()

    def perform_beamsize_measurements(self):
        """
        Run BO-based exploration of the quadrupole strength to get beamsize measurements
        """

        # ignore warnings from UCB generator and Xopt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.create_xopt_object(self.get_vocs("x"))

        # get current value of k
        current_k = self.magnet.bctrl

        # fast scan to get initial guess -- start from current k and scan to the far end of the range
        initial_scan_values = np.linspace(
            current_k,
            self.max_scan_range[0]
            if np.abs(current_k - self.max_scan_range[0])
            > np.abs(current_k - self.max_scan_range[1])
            else self.max_scan_range[1],
            self.n_initial_points,
        )

        try:
            self.X.evaluate_data({"k": initial_scan_values})

            # run iterations for x/y -- ignore warnings from UCB generator
            self.run_iterations("x", self.n_iterations)
            self.run_iterations("y", self.n_iterations)

        except Exception as e:
            raise e
        finally:
            # reset quadrupole strength to original value
            self.magnet.bctrl = current_k

    def _get_beamsizes_scan_values_from_info(self) -> Tuple[np.ndarray]:
        """
        Extract the mean rms beam sizes from the info list, units in meters.
        """
        beam_sizes = []
        for ele in self._info:
            beam_sizes.append(
                np.mean(ele.rms_sizes, axis=0)
                * self.beamsize_measurement.device.resolution
                * 1e-6
            )

        # get scan values and extend for each direction
        scan_values = np.tile(np.array(self.scan_values), (2, 1))

        beam_sizes = np.array(beam_sizes).T

        scan_values_cropped = []
        beam_sizes_cropped = []
        dim_names = ["x", "y"]
        for i in range(2):
            # crop the scans using concavity filter and max beam size filter
            cutoff_size = (
                self._get_cutoff_beamsize(dim_names[i])
                * self.beamsize_measurement.device.resolution
                * 1e-6
            )
            sv_cropped, bs_cropped = crop_scan(
                scan_values=scan_values[i],
                beam_sizes=beam_sizes[i],
                cutoff_max=cutoff_size,
                visualize=self.visualize_cropping,
            )
            scan_values_cropped += [sv_cropped]
            beam_sizes_cropped += [bs_cropped]

        return scan_values_cropped, beam_sizes_cropped

    def get_vocs(self, dim_name):
        """
        Utility function to create x/y vocs.

        This function creates a VOCS object for the given dimension name (x or y).
        It sets the objectives to minimize the rms beam size in pixel squared for that dimension.
        It also sets the constraints based on the minimum signal-to-noise ratio
        and the maximum beam size cutoff based on the smallest beam size measured.

        If a transmission measurement is set, it will also add a transmission constraint to the vocs.

        """

        scan_name = f"{dim_name}_rms_px_sq"
        vocs = VOCS(
            variables={"k": self.max_scan_range},
            objectives={scan_name: "MINIMIZE"},
            observables=["x_rms_px_sq", "y_rms_px_sq"],
        )

        if self.X is not None:
            if self.X.data is not None:
                vocs.constraints = {
                    "min_signal_to_noise_ratio": [
                        "GREATER_THAN",
                        self.min_signal_to_noise_ratio,
                    ],
                    scan_name: [
                        "LESS_THAN",
                        (self._get_cutoff_beamsize(dim_name)) ** 2,
                    ],
                }

                if self.transmission_measurement is not None:
                    vocs.constraints["transmission"] = [
                        "GREATER_THAN",
                        self.transmission_measurement_constraint,
                    ]

        return vocs

    def _get_cutoff_beamsize(self, dim_name):
        """
        return the cutoff beam size for the given dimension, returned in pixel scale
        """
        param_name = f"{dim_name}_rms_px_sq"
        min_size = np.nanmin(self.X.data[param_name].to_numpy(dtype="float"))
        return np.max(
            (
                self.beamsize_cutoff_max * np.sqrt(min_size),
                self.min_beamsize_cutoff / self.beamsize_measurement.device.resolution,
            )
        )


class GPSRMLQuadScanEmittance(MLQuadScanEmittance):
    max_pixels: PositiveInt = 1e5
    n_epochs: PositiveInt = 500
    n_particles: PositiveInt = 10000
    beam_fraction: PositiveFloat = 1.0
    visualize_gpsr: bool = False
    n_stds: PositiveFloat = 5.0
    save_name: str = "gpsr_result"
    image_min_signal_to_noise_ratio: PositiveFloat = 20.0
    median_filter_size: PositiveInt = 3

    def calculate_emittance(self):
        """
        Modify the emittance calculation for the quad scan to utilize GPSR.
        """
        # get the emittance measurement result object
        initial_result = super().calculate_emittance()

        try:
            processed_data = process_automatic_emittance_measurement(
                initial_result,
                n_stds=self.n_stds,
                max_pixels=self.max_pixels,
                median_filter_size=self.median_filter_size,
            )

            print(type(processed_data["images"]))
            print(processed_data["images"].shape)

            gpsr_result = gpsr_fit_quad_scan(
                processed_data["quad_strengths"],
                processed_data["images"],
                processed_data["energy"],
                processed_data["rmat"],
                processed_data["resolution"],
                self.n_epochs,
                self.beam_fraction,
                n_particles=self.n_particles,
                design_twiss=processed_data["design_twiss"],
                visualize=self.visualize_gpsr,
                save_location=self.save_location,
                save_name=self.save_name,
            )

            emittance = np.array(
                [
                    gpsr_result["emittance_x"],
                    gpsr_result["emittance_y"],
                ]
            ).reshape(2, 1)
            formatted_result = EmittanceMeasurementResult(
                quadrupole_focusing_strengths=[processed_data["quad_strengths"]] * 2,
                quadrupole_pv_values=[processed_data["quad_pv_values"]] * 2,
                emittance=emittance,
                bmag=gpsr_result["bmag"],
                twiss_at_screen=gpsr_result["twiss_at_screen"],
                rms_beamsizes=gpsr_result["rms_beamsizes"],
                beam_matrix=gpsr_result["beam_matrix"],
                metadata=initial_result.metadata,
            )

            return formatted_result, gpsr_result
        except Exception as e:
            print(
                f"Error occurred during GPSR emittance calculation: {traceback.format_exc()} "
                "Returning normal emittance calculation results."
            )
            return initial_result, None
