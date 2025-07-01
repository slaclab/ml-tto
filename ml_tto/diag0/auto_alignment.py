from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
import numpy as np


def get_local_region(center_point: dict, vocs: VOCS, fraction: float = 0.1) -> dict:
    """
    calculates the bounds of a local region around a center point with side lengths
    equal to a fixed fraction of the input space for each variable

    """
    if not center_point.keys() == set(vocs.variable_names):
        raise KeyError("Center point keys must match vocs variable names")

    bounds = {}
    widths = {
        ele: vocs.variables[ele][1] - vocs.variables[ele][0]
        for ele in vocs.variable_names
    }

    for name in vocs.variable_names:
        bounds[name] = [
            np.max(
                (center_point[name] - widths[name] * fraction, vocs.variables[name][0])
            ),
            np.min(
                (center_point[name] + widths[name] * fraction, vocs.variables[name][1])
            ),
        ]

    return bounds


def run_automatic_alignment(env):
    """
    Runs the automatic alignment optimization process on DIAG0.

    """

    pvs = list(env.corrector_variables.keys())

    vocs = VOCS(variables=env.get_bounds(pvs), objectives={"rms": "MINIMIZE"})
    local_region = get_local_region(
        env.get_variables(vocs.variables.keys()), vocs, 0.05
    )

    def eval(inputs):
        env.set_variables(inputs)

        bpm_observables = []
        for ele in env.bpm_observables:
            if not ("470" in ele or "520" in ele):
                bpm_observables.append(ele)

        bpm_signals = env.get_observables(bpm_observables)

        return {"rms": np.std([bpm_signals[name] for name in bpm_observables])}

    vocs = VOCS(variables=local_region, objectives={"rms": "MINIMIZE"})
    generator = UpperConfidenceBoundGenerator(vocs=vocs, beta=0.1)
    evaluator = Evaluator(function=eval)

    X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator)

    X.evaluate_data(env.get_variables(vocs.variables.keys()))
    X.random_evaluate(2)

    for i in range(20):
        print(i)
        X.step()

    return X
