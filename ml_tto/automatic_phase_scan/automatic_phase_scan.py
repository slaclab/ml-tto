import time
import warnings
import pprint
from typing import Any, List, Optional
from epics import caget, caput
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
from xopt.generators.bayesian import ExpectedImprovementGenerator

from lcls_tools.common.devices.tcav import TCAV
from ml_tto.automatic_emittance.screen_profile import ScreenBeamProfileMeasurement

from scipy.stats import linregress

class MLTCAVPhasing(BaseModel):
    """Bayesian optimization routine for tuning TCAV phase."""

    beamsize_measurement: ScreenBeamProfileMeasurement
    tcav: TCAV

    n_measurement_shots: PositiveInt = 1
    wait_time: PositiveFloat = 2.0
    scan_values: list[float] = []
    centroids : list[float] = []
    intensities : list[float] = []
    _info: list = []

    n_initial_points: PositiveInt = 3
    n_iterations: PositiveInt = 10

    X: Optional[Xopt] = None

    name: str = "automatic_phase_scan"
    nominal_centroid: Optional[float] = None
    max_scan_range: list[float] = [0, 180]

    verbose: bool = False
    visualize_bo: bool = True
    visualize_cropping: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def optimized_phase(self):
        if self.X.data:
            try:
                return float(self.X.vocs.select_best(self.X.data)[2]["phase"])
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print('Optimized value not yet determined')
            return None
        
    def run(self):
        # acquire the beam posisition without the TCAV on
        self.nominal_centroid = self.acquire_nominal_centroid()

        if self.verbose:
            print(f"nominal centroid: {self.nominal_centroid}")

        # create xopt object
        self.X = self.create_xopt_object()

        # get origonal values
        start_amp = self.tcav.amp_set
        start_phase = self.tcav.phase_set

        if self.verbose:
            print(f"start amp", start_amp)
            print("start phase", start_phase)

        # run optimization - if an error is raised, reset the scan values
        try:
            # initial coarse scan
            end_value = (
                self.max_scan_range[0]
                if abs(start_phase - self.max_scan_range[0]) > abs(start_phase - self.max_scan_range[1])
                else self.max_scan_range[1]
            )
            initial_scan_values = np.linspace(start_phase, end_value, self.n_initial_points)

            if self.verbose:
                print(f"initial scan values: {initial_scan_values}")
            self.X.evaluate_data({"phase": initial_scan_values})
            # run optimization
            for i in range(self.n_iterations):
                if self.verbose:
                    print(f"step:{i}")
                self.X.step()
  
            final_phase = float(self.X.vocs.select_best(self.X.data)[2]["phase"])
            if self.verbose:
                print(f"setting final phase to {final_phase}")
            
            #caput(
            #    "TCAV:DIAG0:11:PREQ", 
            ##    final_phase
            #)
            self.tcav.phase_set = final_phase

        except Exception as e:
            self.tcav.phase_set = start_phase
            raise e
            
        finally:
            self.tcav.amp_set
        

    def create_xopt_object(self):
        """Instantiate Xopt optimizer object."""
        vocs = VOCS(
            variables={"phase": self.max_scan_range},
            constraints={"signal_to_noise_X": ["GREATER_THAN", 4.0],
                         "signal_to_noise_Y": ["GREATER_THAN", 4.0]},
            objectives={"offset": "MINIMIZE"}
        )
        evaluator = Evaluator(function=self._evaluate)
        generator = ExpectedImprovementGenerator(vocs=vocs)
        return Xopt(vocs=vocs, evaluator=evaluator, generator=generator)


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

    def acquire_nominal_centroid(self) -> float:
        """Get centroid without TCAV streaking influence."""
        starting_amplitude = self.tcav.amp_set
        self.tcav.amp_set = 0.0
        time.sleep(self.wait_time)
        result = self.beamsize_measurement.measure(self.n_measurement_shots)
        self.tcav.amp_set = starting_amplitude
        time.sleep(self.wait_time)
        return result.centroids[0,0]

    def _evaluate(self, inputs: dict[str, Any]) -> dict[str, float]:
        """Evaluate the objective function for Bayesian optimization."""
        if self.verbose:
            pprint.pprint(inputs)

        self.tcav.phase_set = inputs["phase"]
        time.sleep(self.wait_time)

        if self.verbose:
            print(f"TCAV Phase set to {inputs['phase']} degrees")

        result = self.beamsize_measurement.measure(self.n_measurement_shots)
        signal_to_noise_X = result.signal_to_noise_ratios[0,0]
        signal_to_noise_Y = result.signal_to_noise_ratios[0,1]
        offset = (self.nominal_centroid - result.centroids[0,0])**2

        return {
            "offset": offset,
            "signal_to_noise_X": signal_to_noise_X,
            "signal_to_noise_Y": signal_to_noise_Y,            
            "centroid": result.centroids[0,0]
        }