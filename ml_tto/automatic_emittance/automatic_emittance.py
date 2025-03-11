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

from pydantic import PositiveInt, SerializeAsAny, field_validator
import time

from ml_tto.automatic_emittance.utils import compute_emit_bmag_machine_units
from ml_tto.automatic_emittance.screen_profile import (
    ScreenBeamProfileMeasurement,
)
from ml_tto.automatic_emittance.scan_cropping import crop_scan


class MLQuadScanEmittance(QuadScanEmittance):
    scan_values: Optional[list[float]] = []
    beamsize_measurement: SerializeAsAny[ScreenBeamProfileMeasurement]
    n_initial_samples: PositiveInt = 3
    n_iterations: PositiveInt = 5
    max_scan_range: Optional[list[float]] = [-10.0, 10.0]
    X: Optional[Xopt] = None

    min_signal_to_noise_ratio: float = 4.0
    n_interpolate_points: Optional[PositiveInt] = 3
    n_grid_points: PositiveInt = 100
    beamsize_cutoff_max: float = 3.0
    beta: float = 10000.0
    visualize_bo: bool = False
    visualize_cropping: bool = False
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

    def measure(self):
        """
        Conduct quadrupole scan to measure the beam phase space.

        Returns:
        -------
        result : EmittanceMeasurementResult
            Object containing the results of the emittance measurement
        """

        # reset the scan values and info
        self.reset()

        # scan magnet strength and measure beamsize
        self.perform_beamsize_measurements()

        return self.calculate_emittance()

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

        # replace last element of info with validated result
        validated_result = fit_result
        self._info[-1] = validated_result

        # collect results
        rms_x = validated_result.rms_sizes[:, 0] / 100
        rms_y = validated_result.rms_sizes[:, 1] / 100

        results = {
            "scaled_x_rms_px_sq": rms_x**2,
            "scaled_y_rms_px_sq": rms_y**2,
            "min_signal_to_noise_ratio": np.min(
                validated_result.signal_to_noise_ratios
            ),
        }
        if self.verbose:
            print(f"Results: {results}")

        return results

    def create_xopt_object(self, vocs):
        evaluator = Evaluator(function=self._evaluate)
        generator = UpperConfidenceBoundGenerator(
            vocs=vocs,
            beta=self.beta,
            numerical_optimizer=GridOptimizer(
                n_grid_points=self.n_grid_points
            ),
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
        # get current value of k
        current_k = self.magnet.bctrl

        # ignore warnings from UCB generator and Xopt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.create_xopt_object(self.get_vocs("x"))

        # evaluate the current point
        self.X.evaluate_data({"k": current_k})

        # fast scan to get initial guess
        self.X.evaluate_data({"k": np.linspace(*self.max_scan_range, 5)})

        # run iterations for x/y -- ignore warnings from UCB generator
        self.run_iterations("x", self.n_iterations)
        self.run_iterations("y", self.n_iterations)

        # reset quadrupole strength to original value
        self.magnet.bctrl = current_k

    def calculate_emittance(self):
        """
        Run the emittance fit using the measured beam sizes and quadrupole strengths.
        """

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
                    [
                        self.design_twiss["beta_x"],
                        self.design_twiss["alpha_x"],
                    ],
                    [
                        self.design_twiss["beta_y"],
                        self.design_twiss["alpha_y"],
                    ],
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
            "twiss_design": (
                twiss_betas_alphas if twiss_betas_alphas is not None else None
            ),
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

        scan_values_cropped = []
        beam_sizes_cropped = []
        for i in range(2):
            # crop the scans using concavity filter and max beam size filter
            sv_cropped, bs_cropped = crop_scan(
                scan_values=scan_values[i],
                beam_sizes=beam_sizes[i],
                cutoff_max=self.beamsize_cutoff_max,
                visualize=self.visualize_cropping,
            )
            scan_values_cropped += [sv_cropped]
            beam_sizes_cropped += [bs_cropped]

        return scan_values_cropped, beam_sizes_cropped

    def get_vocs(self, dim_name):
        """utility function to create x/y vocs"""

        scan_name = f"scaled_{dim_name}_rms_px_sq"
        vocs = VOCS(
            variables={"k": self.max_scan_range},
            objectives={scan_name: "MINIMIZE"},
            observables=["scaled_x_rms_px_sq", "scaled_y_rms_px_sq"],
        )

        if self.X is not None:
            if self.X.data is not None:
                min_size = np.nanmin(
                    self.X.data[scan_name].to_numpy(dtype="float")
                )
                vocs.constraints = {
                    "min_signal_to_noise_ratio": [
                        "GREATER_THAN",
                        self.min_signal_to_noise_ratio,
                    ],
                    scan_name: ["LESS_THAN", self.beamsize_cutoff_max**2 * min_size],
                }

        return vocs
