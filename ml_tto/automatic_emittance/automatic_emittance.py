from copy import deepcopy
import os
import warnings
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.utils import get_local_region
from xopt.numerical_optimizer import GridOptimizer
from lcls_tools.common.measurements.emittance_measurement import QuadScanEmittance
from lcls_tools.common.image.roi import ROI
from typing import Optional

from pydantic import PositiveInt, field_validator

from lcls_tools.common.measurements.screen_profile import (
    ScreenBeamProfileMeasurement,
)
from ml_tto.automatic_emittance.utils import validate_beamsize_measurement_result
from ml_tto.saver import H5Saver
from datetime import datetime
import time


class MLQuadScanEmittance(QuadScanEmittance):
    scan_values: Optional[list[float]] = []
    n_initial_samples: PositiveInt = 3
    n_iterations: PositiveInt = 5
    max_scan_range: Optional[list[float]] = None
    xopt_object: Optional[Xopt] = None

    bounding_box_factor: float = 2.0
    min_log10_intensity: float = 3.0

    @field_validator("beamsize_measurement", mode="after")
    def validate_beamsize_measurement(cls, v, info):
        # check to make sure the the beamsize measurement screen has a region of interest
        # (also requires ScreenBeamProfileMeasurement)
        if not isinstance(v, ScreenBeamProfileMeasurement):
            raise ValueError(
                "Beamsize measurement must be a ScreenBeamProfileMeasurement for MLQuadScanEmittance"
            )

        # check to make sure the the beamsize measurement screen has a region of interest
        if not isinstance(v.image_processor.roi, ROI):
            raise ValueError(
                "Beamsize measurement screen must have a region of interest"
            )
        return v

    def _evaluate(self, inputs):
        # set quadrupole strength
        self.magnet.bctrl = inputs["k"]
        self.scan_values.append(inputs["k"])

        # start by waiting one refesh cycle for bctrl
        # then wait for bact to match bctrl
        time.sleep(0.1)
        while abs(self.magnet.bctrl - self.magnet.bact) > 0.01:
            time.sleep(0.1)

        # make beam size measurement
        self.measure_beamsize()
        fit_result = self._info[-1]

        # validate the beamsize measurement
        validated_result, bb_penalty, log10_total_intensity = (
            validate_beamsize_measurement_result(
                fit_result,
                self.beamsize_measurement.image_processor.roi,
                n_stds=self.bounding_box_factor,
                min_log10_intensity=self.min_log10_intensity,
            )
        )

        # replace last element of info with validated result
        self._info[-1] = validated_result

        # collect results
        results = {
            "bb_penalty": bb_penalty,
            "log10_total_intensity": log10_total_intensity,
            "x_rms_px": validated_result.rms_sizes[:, 0],
            "y_rms_px": validated_result.rms_sizes[:, 1],
        }

        return results

    def perform_beamsize_measurements(self):
        """
        Run BO-based exploration of the quadrupole strength to get beamsize measurements
        """
        # define the optimization problem
        k_range = self.max_scan_range if self.max_scan_range is not None else [-10, 10]
        vocs = VOCS(
            variables={"k": k_range},
            objectives={"x_rms_px": "MINIMIZE"},
            constraints={
                "bb_penalty": ["LESS_THAN", 0.0],
                "log10_total_intensity": ["GREATER_THAN", self.min_log10_intensity],
            },
        )

        self.scan_values = []

        evaluator = Evaluator(function=self._evaluate)

        # ignore warnings from UCB generator and Xopt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generator = ExpectedImprovementGenerator(
                vocs=vocs,
                # beta=100,
                numerical_optimizer=GridOptimizer(n_grid_points=100),
                n_interpolate_points=5,
                n_monte_carlo_samples=64,
            )

            X = Xopt(vocs=vocs, evaluator=evaluator, generator=generator)

            # get local region around current value and make some samples
            local_region = get_local_region({"k": self.magnet.bctrl}, vocs)
            X.random_evaluate(self.n_initial_samples, custom_bounds=local_region)

            # run iterations for x -- ignore warnings from UCB generator
            for i in range(self.n_iterations):
                X.step()

            # run iterations for y
            new_vocs = deepcopy(vocs)
            new_vocs.objectives = {"y_rms_px": "MINIMIZE"}

            X.vocs = new_vocs
            X.generator.vocs = new_vocs

            for i in range(self.n_iterations):
                X.step()

        self.xopt_object = X

    def measure(self):
        """
        Modify the result of the emittance measurement to include the images
        """
        result = super().measure()
        result_dict = result.model_dump()

        result_dict["image_data"] = [ele.model_dump() for ele in self._info]

        if self.save_location is not None:
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            saver = H5Saver()
            saver.save_to_h5(
                result_dict,
                os.path.join(self.save_location, f"emittance_{current_datetime}.h5"),
            )

        return result
