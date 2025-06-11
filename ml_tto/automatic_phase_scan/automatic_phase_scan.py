
import warnings
from typing import Any, List, Optional
import numpy as np
from numpy import ndarray
import torch
from pydantic import (
    ConfigDict,
    PositiveInt,
    SerializeAsAny,
    field_validator,
    PositiveFloat,
    BaseModel
)
from lcls_tools.common.data.model_general_calcs import get_optics
from lcls_tools.common.measurements.measurement import Measurement
from lcls_tools.common.measurements.utils import NDArrayAnnotatedType
from lcls_tools.common.measurements.screen_profile import ScreenBeamProfileMeasurement
from lcls_tools.common.devices.tcav import TCAV
import warnings
import numpy as np
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.numerical_optimizer import GridOptimizer
from typing import Optional, Tuple
import time
from ml_tto.automatic_emittance.scan_cropping import crop_scan
import pprint
class MLTCAVPhasing(BaseModel):

    beamsize_measurement: ScreenBeamProfileMeasurement
    n_measurement_shots: PositiveInt = 1
    _info: Optional[list] = []
    wait_time: PositiveFloat = 1.0
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scan_values: Optional[list[float]] = []
    n_initial_points: PositiveInt = 3
    n_iterations: PositiveInt = 5
    X: Optional[Xopt] = None
    
    verbose: bool = False
    visualize_bo: bool = True
    visualize_cropping: bool = False

    name: str = "automatic_phase_scan"
    tcav: TCAV
    nominal_centroid: Optional[float] = None
    nominal_amplitude: Optional[float] = .135
    max_scan_range: Optional[list[float]] = [0,360]


# well, the difficulty is in putting it all in a python script that uses a config file and stores
# the data properly, use the automatic emittance object in ml-tto
# (https://github.com/slaclab/ml-tto/blob/main/ml_tto/automatic_emittance/automatic_emittance.py)
# / slash the automatic emittance badger env as a guide for creating an MLTCAVPhasing object that
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



    def acquire_nominal_centroid(self, nominal_amplitude):
        '''get centroid without tcav affecting it'''
        starting_amplitude = self.tcav.amp_set
        self.tcav.amp_set = nominal_amplitude
        time.sleep(self.wait_time)
        result = self.beamsize_measurement.measure(self.n_measurement_shots)
        self.tcav.amp_set =  starting_amplitude 
        self.nominal_centroid = result.centroids
        return result.centroids[0]


    def _evaluate(self, inputs):
        print('INPUTS dict')
        pprint.pprint(inputs)
        # where does inputs come from?
        if self.verbose:
            print(f"Setting TCAV Phase to {inputs['phase']} degrees")

        self.tcav.phase_set = inputs['phase']
        self.scan_values.append(inputs['phase'])
        time.sleep(.1)
        
        if self.verbose:
            print(f"TCAV Phase is at {inputs['phase']} degrees")

        self.measure_beamsize()
        fit_result = self._info[-1]
        # replace last element of info with validated result
        # ?
        #ignore multishot stuff for awhile
        validated_result = fit_result
        self._info[-1] = validated_result

        centroid_x = validated_result.centroids[0]
        intensity = float(validated_result.total_intensities)
        print(intensity)
        #pprint.pprint(validated_result)

        if self.nominal_centroid is not None:
            nominal_centroid = self.nominal_centroid[0]
        else:
            nominal_centroid = self.acquire_nominal_centroid(self.nominal_amplitude)


        # where does constraint go?
        # need something that is like..... better way for unpacking and pack f val and constraints.
        offset = np.sqrt((nominal_centroid-centroid_x)**2)
        return {"f": offset,"min_intensity" : intensity}

    






    # MEASUREMENTS


    def measure_beamsize(self):
        """
        Take measurement from measurement device,
        and store results in `self._info`
        """
        time.sleep(self.wait_time)

        result = self.beamsize_measurement.measure(self.n_measurement_shots)
        self._info += [result]



    def perform_beamsize_measurements(self):
        """
        Run BO-based exploration of the quadrupole strength to get beamsize measurements
        """
        ## need to test this and evaluator
        # get current value of phase
        current_phase = self.tcav.phase_set

        # ignore warnings from UCB generator and Xopt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #self.create_xopt_object()

        # fast scan to get initial guess -- start from current k and scan to the far end of the range
        initial_scan_values = np.linspace(
            current_phase,
            self.max_scan_range[0]
            if np.abs(current_phase - self.max_scan_range[0])
            > np.abs(current_phase - self.max_scan_range[1])
            else self.max_scan_range[1],
            self.n_initial_points,
        )

        self.X.evaluate_data({"phase": initial_scan_values})

        self.run_iterations(self.n_iterations)

        self.tcav.phase_set = current_phase






    # XOPT 

    def create_xopt_object(self, vocs):
        evaluator = Evaluator(function=self._evaluate)
        generator = UpperConfidenceBoundGenerator(vocs=vocs)
        # for now
        generator.gp_constructor.use_low_noise_prior = True
        self.X = Xopt(vocs=vocs, evaluator=evaluator, generator=generator)
        print(self.X)

    def update_xopt_object(self, vocs):
        self.X.vocs = vocs
        self.X.generator.vocs = vocs

    def reset(self):
        self.scan_values = []
        self._info = []

    def run_iterations(self, n_iterations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for _ in range(n_iterations):

                if self.visualize_bo:
                    self.X.generator.train_model()
                    self.X.generator.visualize_model(
                        exponentiate=True,
                        show_feasibility=True,
                    )

                self.X.step()





    def fit_nominal_amplitude(self,data):
        pass
