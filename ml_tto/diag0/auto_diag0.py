from ml_tto.diag0.auto_6d import run_automatic_6d_measurement
from ml_tto.diag0.auto_alignment import run_automatic_alignment
from ml_tto.diag0.auto_tcav_phasing import run_automatic_tcav_phasing


def run_automatic_diag0(env, save_filename):
    """
    Runs the automatic diagnostic procedures on DIAG0 for 6D phase space measurement.

    Parameters:
        env (Environment): The environment in which the measurements are performed.
        save_filename (str): The filename where the results are saved.

    Returns:
        dict: A dictionary containing the results of the automatic measurements.
    """

    # get initial state of the environment
    initial_state = env.get_variables(env.variables.keys())

    try:
        # run automatic alignment to OTRDG04
        env.tcav.amp_set = 0.0  # turn off the TCAV
        env.remove_screen("OTRDG02") # remove OTRDG02 screen

        alignment_result = run_automatic_alignment(env, to_screen_name="OTRDG04")

        # run automatic TCAV phasing
        env.tcav.amp_set = env.tcav_on_amp  # ensure TCAV is set to the nominal value before phasing
        tcav_phasing_result = run_automatic_tcav_phasing(env)

        # run automatic 6D measurement
        six_d_measurement_result = run_automatic_6d_measurement(env, save_filename)

        # collect results
        results = {
            "alignment": alignment_result,
            "tcav_phasing": tcav_phasing_result,
            "six_d_measurement": six_d_measurement_result,
        }

        return results

    except Exception as e:
        # restore the initial state if there is an error
        env.set_variables(initial_state)
        raise e
