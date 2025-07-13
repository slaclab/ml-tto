import time
import torch
import pprint
from typing import Any, Optional
from epics import caget
import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from gpytorch.kernels import CosineKernel, ScaleKernel
from gpytorch.priors import GammaPrior
import numpy as np
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import (
    ExpectedImprovementGenerator, UpperConfidenceBoundGenerator
)
from xopt.generators.bayesian.models.standard import StandardModelConstructor

from lcls_tools.common.devices.tcav import TCAV
from ml_tto.automatic_emittance.transmission import TransmissionMeasurement
from lcls_tools.common.devices.bpm import BPM
from lcls_tools.common.devices.reader import create_bpm

from scipy.stats import linregress


class MLTCAVPhasing(BaseModel):
    """Bayesian optimization routine for tuning TCAV phase."""

    tcav: Any
    bpm: BPM
    transmission_measurement: TransmissionMeasurement

    n_measurement_shots: PositiveInt = 1
    wait_time: PositiveFloat = 2.0

    n_initial_points: PositiveInt = 5
    n_iterations: PositiveInt = 10

    X: Optional[Xopt] = None

    name: str = "automatic_phase_scan"
    nominal_centroid: Optional[float] = None
    max_scan_range: list[float] = [50, 150]

    verbose: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def optimized_phase(self):
        if self.X.data:
            try:
                return float(self.X.vocs.select_best(self.X.data)[2]["phase"])
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Optimized value not yet determined")
            return None

    def run(self):
        # make sure that the tcav is in accel mode
        if caget(self.tcav.controls_information.control_name + ":MODECFG") != 1:
            raise RuntimeError("tcav must be in ACCEL mode config")

        # acquire the beam posisition without the TCAV on
        self.nominal_centroid = self.acquire_nominal_centroid()

        if self.verbose:
            print(f"nominal centroid: {self.nominal_centroid}")

        # create xopt object
        self.X = self.create_xopt_object()

        # get origonal values
        start_amp = self.tcav.amplitude
        start_phase = self.tcav.phase

        if self.verbose:
            print("start amp", start_amp)
            print("start phase", start_phase)

        # run optimization - if an error is raised, reset the scan values
        try:
            # initial coarse scan
            initial_scan_values = np.linspace(
                start_phase*0.9, start_phase*1.1, self.n_initial_points
            )

            # evaluate current point
            self.X.evaluate_data({"phase": start_phase})
            if self.X.data.min()["offset"] < 1e-2:
                print("converged")
                return self.X
            
            if self.verbose:
                print(f"initial scan values: {initial_scan_values}")
            self.X.evaluate_data({"phase": initial_scan_values})

            # run optimization
            for i in range(self.n_iterations):
                if self.X.data.min()["offset"] < 1e-2:
                    print("converged")
                
                if self.verbose:
                    print(f"step:{i}")
                self.X.step()

            final_phase = float(self.X.vocs.select_best(self.X.data)[2]["phase"])
            if self.verbose:
                print(f"setting final phase to {final_phase}")

            self.tcav.phase = final_phase

        except Exception as e:
            self.tcav.phase = start_phase
            raise e

        finally:
            self.tcav.amplitude = start_amp

    def create_xopt_object(self):
        """Instantiate Xopt optimizer object."""
        vocs = VOCS(
            variables={"phase": self.max_scan_range},
            objectives={"offset": "MINIMIZE"},
            constraints={"transmission":["GREATER_THAN", 0.9]}
        )
        evaluator = Evaluator(function=self._evaluate)

        class OffsetPrior(torch.nn.Module):
            def forward(self, X):
                return 100*torch.ones_like(X).squeeze(dim=-1)
                
        gp_constructor = StandardModelConstructor(
            mean_modules={"offset": OffsetPrior()},
            use_low_noise_prior=True
        )
        generator = UpperConfidenceBoundGenerator(
            vocs=vocs, gp_constructor = gp_constructor
        )
        return Xopt(vocs=vocs, evaluator=evaluator, generator=generator)

    def acquire_nominal_centroid(self) -> float:
        """Get centroid without TCAV streaking influence."""
        starting_amplitude = self.tcav.amplitude
        self.tcav.amplitude = 0.0
        time.sleep(self.wait_time)

        result = self.bpm.x

        self.tcav.amplitude = starting_amplitude
        time.sleep(self.wait_time)

        return result

    def _evaluate(self, inputs: dict[str, Any]) -> dict[str, float]:
        """Evaluate the objective function for Bayesian optimization."""
        if self.verbose:
            pprint.pprint(inputs)

        self.tcav.phase = inputs["phase"]
        time.sleep(self.wait_time)

        if self.verbose:
            print(f"TCAV Phase set to {inputs['phase']} degrees")

        transmission = self.transmission_measurement.measure()["transmission"]
        if transmission > 0.8:
            offset = (self.nominal_centroid - self.bpm.x) ** 2
            centroid = self.bpm.x
        else:
            offset = np.nan
            centroid = np.nan

        result = {"offset": offset,"centroid": centroid, "transmission": transmission}

        return result


def run_automatic_tcav_phasing(env):
    tcav = env.tcav
    transmission_measurement = TransmissionMeasurement(
        upstream_bpm = env.upstream_bpm,
        downstream_bpm = env.downstream_bpm
    )
    
    phaser = MLTCAVPhasing(
        bpm=env.downstream_bpm, tcav=tcav, transmission_measurement=transmission_measurement, wait_time=1.0, verbose=True
    )

    phaser.run()

    return phaser.X
