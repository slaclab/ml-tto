import logging
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.objectives import CustomXoptObjective
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from botorch.exceptions.errors import OptimizationGradientError
import numpy as np
import torch
from gpytorch.kernels import ScaleKernel, LinearKernel
import traceback

# Setup Logging 
logger = logging.getLogger("auto_alignment")


def get_local_region(center_point: dict, vocs: VOCS, fraction: float = 0.1) -> dict:
    """
    calculates the bounds of a local region around a center point with side lengths
    equal to a fixed fraction of the input space for each variable

    """
    logger.debug("Calculating local region bounds.")
    if not center_point.keys() == set(vocs.variable_names):
        logger.error("Center point keys must match VOCS variable names")
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

    logger.debug(f"Local region: {bounds}")
    return bounds


alignment_pvs = {
    "OTRDG02": {
        "corrector_pvs": [
            f"XCOR:DIAG0:{ele}:BCTRL" for ele in [178, 218, 280, 290, 340]
        ]
        + [f"YCOR:DIAG0:{ele}:BCTRL" for ele in [199, 247, 280, 290, 340]],
        "bpms": [
            f"BPMS:DIAG0:{ele}:XSCDTH" for ele in [210, 230, 270, 285, 330, 370, 390]
        ]
        + [f"BPMS:DIAG0:{ele}:YSCDTH" for ele in [210, 230, 270, 285, 330, 370, 390]],
    },
    "OTRDG04": {
        "corrector_pvs": [
            f"XCOR:DIAG0:{ele}:BCTRL" for ele in [178, 218, 280, 290, 340, 380, 460]
        ]
        + [f"YCOR:DIAG0:{ele}:BCTRL" for ele in [199, 247, 280, 290, 340, 380, 460]],
        "bpms": [
            f"BPMS:DIAG0:{ele}:XSCDTH" for ele in [270, 285, 330, 370, 390, 470, 520]
        ]
        + [f"BPMS:DIAG0:{ele}:YSCDTH" for ele in [270, 285, 330, 370, 390, 470, 520]],
    },
}


def run_automatic_alignment(
    env, to_screen_name="OTRDG04", n_steps=20, old_data=None, target_value=1.0
):
    """
    Runs the automatic alignment optimization process on DIAG0 to
    `to_screen_name`.

    Parameters:
        env (Environment): The environment in which the optimization is performed.
        to_screen_name (str): The name of the screen to align to. Default is "OTRDG02".

    """
    logger.info(f"Starting automatic alignment for screen: {to_screen_name}")
    # if just transporting beam to OTRDG02, use all BPMs except 470 and 520
    pvs = alignment_pvs[to_screen_name]["corrector_pvs"]
    bpm_observables = alignment_pvs[to_screen_name]["bpms"]

    temp_vocs = VOCS(variables=env.get_bounds(pvs), observables=[])
    local_region = get_local_region(
        env.get_variables(temp_vocs.variables.keys()), temp_vocs, 0.15
    )

    def eval(inputs):
        try:
            env.set_variables(inputs)
        except RuntimeError:
            logger.warning("Error while setting variables.")
            # transmission below 0.8
            norm = np.nan
            bpm_signals = {name: np.nan for name in bpm_observables}
            transmission = 0.5
            return {"norm": norm, "transmission": transmission} | bpm_signals

        transmission = env.get_observables(["transmission"])["transmission"]
        try:
            bpm_signals = env.get_observables(bpm_observables)
            norm = np.linalg.norm([bpm_signals[name] for name in bpm_observables])
        except KeyError:
            logger.warning("Error while getting observables")
            norm = np.nan
            bpm_signals = {name: np.nan for name in bpm_observables}

        # pop input keys from bpm_signals
        for name in inputs.keys():
            if name in bpm_signals:
                bpm_signals.pop(name)

        return {"norm": norm, "transmission": transmission} | bpm_signals

    vocs = VOCS(
        variables=local_region,
        observables=bpm_observables,
        constraints={"transmission": ["GREATER_THAN", 0.9]},
    )

    # create custom objective
    class MyObjective(CustomXoptObjective):
        def forward(self, samples, X=None):
            return -torch.norm(
                torch.stack(
                    [
                        samples[..., self.vocs.observable_names.index(name)]
                        for name in bpm_observables
                    ]
                ),
                dim=0,
            )

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
    generator.numerical_optimizer.max_time = 2.5

    evaluator = Evaluator(function=eval)

    X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator, strict=False)

    logger.info("Starting evaluation")
    # evaluate
    X.evaluate_data(env.get_variables(vocs.variables.keys()))
    if X.data.min()["norm"] < target_value:
        print("converged")
        return X

    random_sample_region = get_local_region(
        env.get_variables(vocs.variables.keys()), X.vocs, fraction=0.1
    )

    if old_data is not None:
        logger.info("Adding old data.")
        X.add_data(old_data)
    else:
        logger.info("Generating random points.")
        X.random_evaluate(10, custom_bounds=random_sample_region)

    try:
        for i in range(n_steps):
            logger.info(f"At step {i}")
            if X.data.min()["norm"] < target_value:
                logger.info("Converged")
                break

            # try running a bo step until we succeed -- max 5 tries
            for _ in range(5):
                try:
                    X.step()
                    break
                except OptimizationGradientError:
                    logger.warning("gradient error, adding random evals and then trying again")
                    X.random_evaluate(1)

    except Exception:
        logger.error("Exception:")
        logger.error(traceback.format_exc())
    finally:
        logger.info("evaluating the best point")
        X.evaluate(
            X.data[X.vocs.variable_names].iloc[X.data.idxmin()["norm"]].to_dict()
        )
        return X
