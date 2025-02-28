from copy import deepcopy
import warnings

import numpy as np
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.numerical_optimizer import GridOptimizer
from lcls_tools.common.measurements.emittance_measurement import (
    QuadScanEmittance,
    EmittanceMeasurementResult,
)
from lcls_tools.common.image.roi import ROI
from lcls_tools.common.data.emittance import compute_emit_bmag
from lcls_tools.common.data.model_general_calcs import bdes_to_kmod, get_optics
from typing import Optional, Tuple

from pydantic import PositiveInt, field_validator
import time

from ml_tto.automatic_emittance.utils import compute_emit_bmag_machine_units
from ml_tto.automatic_emittance.screen_profile import ScreenBeamProfileMeasurement


class MLQuadScanEmittance(QuadScanEmittance):
    scan_values: Optional[list[float]] = []
    n_initial_samples: PositiveInt = 3
    n_iterations: PositiveInt = 5
    max_scan_range: Optional[list[float]] = None
    xopt_object: Optional[Xopt] = None

    bounding_box_factor: float = 2.0
    min_log10_intensity: float = 3.0
    cutoff_max: float = 3.0
    visualize_bo: bool = False
    verbose: bool = False

    @field_validator("beamsize_measurement", mode="after")
    def validate_beamsize_measurement(cls, v, info):
        # check to make sure the the beamsize measurement screen has a region of interest
        # (also requires ScreenBeamProfileMeasurement)
        if not isinstance(v, ScreenBeamProfileMeasurement):
            raise ValueError(
                "Beamsize measurement must be a ScreenBeamProfileMeasurement for MLQuadScanEmittance"
            )

        return v
        # check to make sure the the beamsize measurement screen has a region of interest
        # if not isinstance(v.image_processor.roi, ROI):
        #    raise ValueError(
        #        "Beamsize measurement screen must have a region of interest"
        #    )
        # return v

    def _evaluate(self, inputs):
        # set quadrupole strength
        if self.verbose:
            print(f"Setting quadrupole strength to {inputs['k']}")
        self.magnet.bctrl = inputs["k"]
        self.scan_values.append(inputs["k"])

        # start by waiting one refesh cycle for bctrl
        # then wait for bact to match bctrl
        # bctrl referesh rate is less than 10 ms
        time.sleep(0.02)
        while abs(self.magnet.bctrl - self.magnet.bact) > 0.01:
            time.sleep(0.05)

        if self.verbose:
            print(f"Quadrupole strength bact is {self.magnet.bact}")

        # make beam size measurement
        self.measure_beamsize()
        fit_result = self._info[-1]

        # # validate the beamsize measurement
        # validated_result, bb_penalty, log10_total_intensity = (
        #     validate_beamsize_measurement_result(
        #         fit_result,
        #         self.beamsize_measurement.image_processor.roi,
        #         n_stds=self.bounding_box_factor,
        #         min_log10_intensity=self.min_log10_intensity,
        #     )
        # )

        # replace last element of info with validated result
        validated_result = fit_result
        self._info[-1] = validated_result

        # collect results
        rms_x = validated_result.rms_sizes[:, 0] / 100
        rms_y = validated_result.rms_sizes[:, 1] / 100

        # replace any nan values with 10.0
        #rms_x[np.isnan(rms_x)] = 10.0
        #rms_y[np.isnan(rms_y)] = 10.0

        results = {
            "scaled_x_rms_px": rms_x,
            "scaled_y_rms_px": rms_y,
            "min_signal_to_noise_ratio": np.min(
                validated_result.signal_to_noise_ratios
            ),
        }
        if self.verbose:
            print(f"Results: {results}")

        return results

    def perform_beamsize_measurements(self):
        """
        Run BO-based exploration of the quadrupole strength to get beamsize measurements
        """
        # define the optimization problem
        k_range = self.max_scan_range if self.max_scan_range is not None else [-10, 10]
        x_vocs = VOCS(
            variables={"k": k_range},
            objectives={"scaled_x_rms_px": "MINIMIZE"},
            observables=["scaled_x_rms_px", "scaled_y_rms_px"],
        )
        y_vocs = deepcopy(x_vocs)
        y_vocs.objectives = {"scaled_y_rms_px": "MINIMIZE"}

        self.scan_values = []

        evaluator = Evaluator(function=self._evaluate)

        # get current value of k
        current_k = self.magnet.bctrl

        # ignore warnings from UCB generator and Xopt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generator = UpperConfidenceBoundGenerator(
                vocs=x_vocs,
                beta=1000.0,
                numerical_optimizer=GridOptimizer(n_grid_points=100),
                n_interpolate_points=3,
                n_monte_carlo_samples=64,
            )

            X = Xopt(vocs=x_vocs, evaluator=evaluator, generator=generator)

            # evaluate the current point
            X.evaluate_data({"k": current_k})

            # get local region around current value and make some samples
            # local_region = get_local_region({"k": current_k}, x_vocs)
            # X.random_evaluate(self.n_initial_samples, custom_bounds=local_region)

            X.evaluate_data({"k": np.linspace(k_range[0], k_range[1], 5)})

            # run iterations for x/y -- ignore warnings from UCB generator
            for _ in range(self.n_iterations):
                min_size = np.nanmin(X.data["scaled_x_rms_px"].to_numpy(dtype="float"))
                x_vocs.constraints = {
                    "min_signal_to_noise_ratio": ["GREATER_THAN", 4],
                    "scaled_x_rms_px": ["LESS_THAN", self.cutoff_max * min_size],
                }

                X.vocs = x_vocs
                X.generator.vocs = x_vocs

                if self.visualize_bo:
                    X.generator.train_model()
                    X.generator.visualize_model(
                        exponentiate=True,
                        show_feasibility=True,
                    )

                X.step()

            X.vocs = y_vocs
            X.generator.vocs = y_vocs
            for _ in range(self.n_iterations):
                min_size = np.nanmin(X.data["scaled_y_rms_px"].to_numpy(dtype="float"))
                y_vocs.constraints = {
                    "scaled_y_rms_px": ["LESS_THAN", self.cutoff_max * min_size]
                }

                X.vocs = y_vocs
                X.generator.vocs = y_vocs

                if self.visualize_bo:
                    X.generator.train_model()
                    X.generator.visualize_model(
                        exponentiate=True,
                        show_feasibility=True,
                    )

                X.step()

        # reset quadrupole strength to original value
        self.magnet.bctrl = current_k

        self.xopt_object = X

    def measure(self):
        """
        Conduct quadrupole scan to measure the beam phase space.

        Returns:
        -------
        result : EmittanceMeasurementResult
            Object containing the results of the emittance measurement
        """

        self._info = []
        # scan magnet strength and measure beamsize
        self.perform_beamsize_measurements()

        # extract beam sizes from info
        scan_values, beam_sizes = self._get_beamsizes_scan_values_from_info()

        # get transport matrix and design twiss values from meme
        # TODO: get settings from arbitrary methods (ie. not meme)
        if self.rmat is None and self.design_twiss is None:
            optics = get_optics(
                self.magnet_name,
                self.device_measurement.device.name,
            )

            self.rmat = optics["rmat"]
            self.design_twiss = optics["design_twiss"]

        magnet_length = self.magnet.metadata.l_eff
        if magnet_length is None:
            raise ValueError(
                "magnet length needs to be specified for magnet "
                f"{self.magnet.name} to be used in emittance measurement"
            )

        # organize data into arrays for use in `compute_emit_bmag`
        # rmat = np.stack([self.rmat[0:2, 0:2], self.rmat[2:4, 2:4]])
        if self.design_twiss:
            twiss_betas_alphas = np.array(
                [
                    [self.design_twiss["beta_x"], self.design_twiss["alpha_x"]],
                    [self.design_twiss["beta_y"], self.design_twiss["alpha_y"]],
                ]
            )

        else:
            twiss_betas_alphas = None

        inputs = {
            "quad_vals": scan_values,
            "beamsizes": beam_sizes,
            "q_len": magnet_length,
            "rmat": self.rmat,
            "energy": self.energy,
            "twiss_design": twiss_betas_alphas
            if twiss_betas_alphas is not None
            else None,
        }

        # Call wrapper that takes quads in machine units and beamsize in meters
        results = compute_emit_bmag_machine_units(**inputs)

        results.update({"metadata": self.model_dump()})

        # collect information into EmittanceMeasurementResult object
        return EmittanceMeasurementResult(**results)

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

        for i in range(2):
            min_observed_size = np.nanmin(beam_sizes[i])

            # cut out any indicies where the beam size is larger than some amount
            mask = beam_sizes[i] > self.cutoff_max * min_observed_size
            beam_sizes[i][mask] = np.nan

        return scan_values, beam_sizes
