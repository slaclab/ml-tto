import logging

logger = logging.getLogger("auto_focusing")

import numpy as np
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import (
    ExpectedImprovementGenerator,
    UpperConfidenceBoundGenerator,
)
from xopt.utils import get_local_region
from botorch.exceptions.errors import OptimizationGradientError
import traceback


def run_auto_focusing(
    env,
    screen_name,
    quads,
    n_steps=20,
    old_data=None,
    target_value=100,
    objective="total_size",
):
    """
    Runs the automatic focusing optimization process on DIAG0 to
    `screen_name`.

    Parameters:
        env (Environment): The environment in which the optimization is performed.
        screen_name (str): The name of the screen to focus on.
        quads (list): List of quadrupole names to use for focusing.
        n_steps (int): Number of steps for the optimization. Default is 20.
        old_data (dict): Previous data to use for optimization. Default is None.

    Returns:
        dict: The results of the optimization.
    """

    # set the screen
    env.set_screen(screen_name)

    # Implementation of auto-focusing logic goes here

    temp_vocs = VOCS(variables=env.get_bounds(quads), observables=[])
    local_region = get_local_region(
        env.get_variables(temp_vocs.variables.keys()), temp_vocs, 0.05
    )

    vocs = VOCS(
        variables=local_region,
        objectives={objective: "MINIMIZE"},
        constraints={"transmission": ["GREATER_THAN", 0.7]},
    )

    def eval(inputs):
        try:
            env.set_variables(inputs)
        except RuntimeError:
            return {objective: np.nan, "transmission": env.bad_transmission}

        results = env.get_observables([objective, "transmission"])
        for name in inputs:
            results.pop(name)

        return results

    evaluator = Evaluator(function=eval)

    generator = ExpectedImprovementGenerator(
        vocs=vocs,
        n_interpolate_points=3,
    )
    generator.numerical_optimizer.max_time = 2.5

    X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator)

    try:
        X.evaluate_data(env.get_variables(vocs.variables.keys()))
        if X.vocs.select_best(X.data)[1] < target_value:
            logger.info(f"converged with value {X.vocs.select_best(X.data)[1]}")
            return X

        random_sample_region = get_local_region(
            env.get_variables(vocs.variables.keys()), X.vocs, fraction=0.1
        )

        if old_data is not None:
            X.add_data(old_data)
        else:
            X.random_evaluate(3, custom_bounds=random_sample_region)

        for i in range(n_steps):
            if X.vocs.select_best(X.data)[1] < target_value:
                logger.info(f"converged with value {X.vocs.select_best(X.data)[1]}")
                break

            # if any of the evaluations are close to the objective value - use turbo
            if (
                np.any(X.data[objective] < target_value * 1.5)
                and X.generator.turbo_controller is None
            ):
                logger.info(
                    "found a point close to the optimum, starting turbo controller"
                )
                X.generator.turbo_controller = "optimize"

            # try running a bo step until we succeed -- max 5 tries
            for _ in range(5):
                try:
                    X.step()
                    break
                except OptimizationGradientError:
                    logger.warning(
                        "gradient error, adding random evals and then trying again"
                    )
                    X.random_evaluate(1)
    except Exception:
        logger.error(traceback.format_exc())
        raise
    finally:
        idx = X.vocs.select_best(X.data)[0]
        result = X.evaluate_data(X.data.iloc[idx][X.vocs.variable_names])
        logger.info(f"evaluated the best point: {objective}={result[objective]}")

        return X
