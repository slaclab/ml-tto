from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.objectives import CustomXoptObjective
from xopt.generators.bayesian.models.standard import StandardModelConstructor
import numpy as np
import torch
from gpytorch.kernels import ScaleKernel, LinearKernel


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


def run_automatic_alignment(env, to_screen_name="OTRDG02", n_steps=20):
    """
    Runs the automatic alignment optimization process on DIAG0 to
    `to_screen_name`.

    Parameters:
        env (Environment): The environment in which the optimization is performed.
        to_screen_name (str): The name of the screen to align to. Default is "OTRDG02".

    """

    pvs = list(env.corrector_variables.keys())

    vocs = VOCS(variables=env.get_bounds(pvs), objectives={"rms": "MINIMIZE"})
    local_region = get_local_region(env.get_variables(vocs.variables.keys()), vocs, 0.1)

    # if just transporting beam to OTRDG02, use all BPMs except 470 and 520
    if to_screen_name == "OTRDG02":
        bpm_observables = []
        for ele in env.bpm_observables:
            if not ("470" in ele or "520" in ele):
                bpm_observables.append(ele)
    else:
        # if aligning to OTRDG04, use all BPMs
        bpm_observables = env.bpm_observables

    def eval(inputs):
        env.set_variables(inputs)

        transmission = env.get_observables(["transmission"])["transmission"]
        try:
            bpm_signals = env.get_observables(bpm_observables)
            rms = np.linalg.norm([bpm_signals[name] for name in bpm_observables])
        except KeyError:
            rms = np.NaN

        return {"rms": rms, "transmission": transmission} | bpm_signals

    vocs = VOCS(
        variables=local_region,
        observables=bpm_observables,
        constraints={"transmission": ["GREATER_THAN", 0.8]},
    )

    # create custom objective
    class MyObjective(CustomXoptObjective):
        def forward(self, samples, X=None):
            return -torch.norm(
                torch.stack(
                    [
                        samples[..., self.vocs.output_names.index(name)]
                        for name in bpm_observables
                    ]
                ),
                dim=0,
            ).log()

    covar_modules = {name: ScaleKernel(LinearKernel()) for name in bpm_observables}
    gp_constructor = StandardModelConstructor(
        covar_modules=covar_modules,
    )

    generator = ExpectedImprovementGenerator(
        vocs=vocs,
        custom_objective=MyObjective(vocs),
        gp_constructor=gp_constructor,
        n_interpolate_points=3,
    )

    evaluator = Evaluator(function=eval)

    X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator)

    X.evaluate_data(env.get_variables(vocs.variables.keys()))
    random_sample_region = get_local_region(
        env.get_variables(vocs.variables.keys()), X.vocs, fraction=0.1
    )
    X.random_evaluate(10, custom_bounds=random_sample_region)

    for i in range(n_steps):
        print(i)
        X.step()

    return X
