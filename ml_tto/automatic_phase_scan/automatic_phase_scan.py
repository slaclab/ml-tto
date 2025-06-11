import time
import warnings
import pprint
from typing import Any, List, Optional

import numpy as np
import torch
from numpy import ndarray
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveFloat,
    PositiveInt
)

from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import UpperConfidenceBoundGenerator

from lcls_tools.common.devices.tcav import TCAV
from lcls_tools.common.measurements.screen_profile import ScreenBeamProfileMeasurement

from ml_tto.automatic_emittance.scan_cropping import crop_scan
from scipy.stats import linregress

class MLTCAVPhasing(BaseModel):
    """Bayesian optimization routine for tuning TCAV phase."""

    beamsize_measurement: ScreenBeamProfileMeasurement
    tcav: TCAV

    n_measurement_shots: PositiveInt = 1
    wait_time: PositiveFloat = 1.0
    scan_values: list[float] = []
    centroids : list[float] = []
    intensities : list[float] = []
    _info: list = []

    n_initial_points: PositiveInt = 3
    n_iterations: PositiveInt = 5

    X: Optional[Xopt] = None
    vocs: Optional[VOCS] = None

    name: str = "automatic_phase_scan"
    nominal_centroid: Optional[float] = None
    nominal_amplitude: float = 0.135
    max_scan_range: list[float] = [0, 360]

    verbose: bool = False
    visualize_bo: bool = True
    visualize_cropping: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


    def determine_streaking(self):
        """Analyze centroid trend to determine streaking direction (not implemented)."""
        slope = self.centroid_slope(self.scan_values, self.centroids)
        if slope > 0:
            print("Centroid is increasing with phase (right streaking)")
        elif slope < 0:
            print("Centroid is decreasing with phase (left streaking)")
        else:
            print("No centroid trend")

    def centroid_slope(self, phases, centroids):
        slope, intercept, r_value, p_value, std_err = linregress(phases, centroids)
        return slope

    def acquire_nominal_centroid(self, nominal_amplitude: float) -> float:
        """Get centroid without TCAV streaking influence."""
        starting_amplitude = self.tcav.amp_set
        self.tcav.amp_set = nominal_amplitude
        time.sleep(self.wait_time)
        result = self.beamsize_measurement.measure(self.n_measurement_shots)
        self.tcav.amp_set = starting_amplitude
        self.nominal_centroid = result.centroids
        return result.centroids[0]

    def _evaluate(self, inputs: dict[str, Any]) -> dict[str, float]:
        """Evaluate the objective function for Bayesian optimization."""
        if self.verbose:
            pprint.pprint(inputs)

        self.tcav.phase_set = inputs["phase"]
        self.scan_values.append(inputs["phase"])
        time.sleep(0.1)

        if self.verbose:
            print(f"TCAV Phase set to {inputs['phase']} degrees")

        self.measure_beamsize()
        validated_result = self._info[-1]

        centroid_x = validated_result.centroids[0,0]
        print(validated_result.centroids)
        print(centroid_x)
        self.centroids.append(centroid_x)

        intensity = float(validated_result.total_intensities)
        self.intensities.append(intensity)

        if self.nominal_centroid is not None:
            nominal_centroid = self.nominal_centroid[0]
        else:
            nominal_centroid = self.acquire_nominal_centroid(self.nominal_amplitude)

        offset = np.abs(nominal_centroid - centroid_x)

        # need a way to unpack constraints better, may not always be min instensity
        return {
            "f": offset,
            "min_intensity": intensity
        }

    def measure_beamsize(self):
        """Take a single beamsize measurement."""
        time.sleep(self.wait_time)
        result = self.beamsize_measurement.measure(self.n_measurement_shots)
        self._info.append(result)

    def perform_beamsize_measurements(self):
        """Perform initial scans and run Bayesian optimization."""
        current_phase = self.tcav.phase_set

        scan_values = (
            self.max_scan_range[0]
            if abs(current_phase - self.max_scan_range[0]) > abs(current_phase - self.max_scan_range[1])
            else self.max_scan_range[1]
        )
        initial_scan_values = np.linspace(current_phase, scan_values, self.n_initial_points)

        self.X.evaluate_data({"phase": initial_scan_values})
        self.run_iterations(self.n_iterations)

        self.tcav.phase_set = current_phase

    def create_xopt_object(self, vocs: VOCS):
        """Instantiate Xopt optimizer object."""
        evaluator = Evaluator(function=self._evaluate)
        generator = UpperConfidenceBoundGenerator(vocs=vocs)
        generator.gp_constructor.use_low_noise_prior = True
        self.X = Xopt(vocs=vocs, evaluator=evaluator, generator=generator)

        if self.verbose:
            print(self.X)

    def update_xopt_object(self, vocs: VOCS):
        """Update VOCS in existing Xopt object."""
        self.X.vocs = vocs
        self.X.generator.vocs = vocs

    def run_iterations(self, n_iterations: int):
        """Run N optimization steps with visualization if enabled."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for _ in range(n_iterations):
                if self.visualize_bo:
                    self.X.generator.train_model()
                    self.X.generator.visualize_model(
                        exponentiate=True,
                        show_feasibility=True
                    )
                self.X.step()

    def reset(self):
        """Clear all scan history and results."""
        self.scan_values = []
        self.centroids = []
        self.intensities = []
        self._info = []



# executes this on the simulated server' (edited) 
# Also reduce the size of your screen diagnostic to simulate it going off the
# side of the screen for certain phases, and include a constraint that is violated when
# there is no beam on the screen (min intensity for example) (edited)
# also we will need to determine a way to know if we are streaking in the right direction
# (ie. positive time to the left, negative time to the right) since there will be minima at
# every multiple of 180
# probably need to calculate the slope of the centroid as a function of phase and
# incorporate that into the objective
#one function that measures zero
#one function that evaluates
#one function does optimization
#one help function that calls it all together
# clean code, make better way for passing constraints

#setup, verify result
#reduce screen size, verify constraint
#determine streaking
#remove noise = False flag