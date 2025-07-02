from ml_tto.diag0.auto_emittance import run_automatic_emittance

from ml_tto.saver import H5Saver

def run_automatic_6d_measurement(env, save_filename):
    """
    Does the following:
    1. Insert OTRDG02
    2. Run automatic emittance measurement with TCAV off.
    4. Run automatic emittance measurement with TCAV on.
    5. Remove OTRDG02
    6. Run automatic emittance measurement with TCAV off.
    7. Run automatic emittance measurement with TCAV on.

    """

    # insert OTRDG02
    env.insert_screen("OTRDG02")

    # turn off TCAV
    env.tcav.amp_set = 0.0

    data = {}

    # run automatic emittance measurement with TCAV off
    emittance_result_OTRDG02_off, _, X = run_automatic_emittance(env, "OTRDG02")
    data["OTRDG02_off"] = emittance_result_OTRDG02_off

    # turn on TCAV
    env.tcav.amp_set = env.tcav_on_amp

    # run automatic emittance measurement with TCAV on
    emittance_result_OTRDG02_on, _, X = run_automatic_emittance(env, "OTRDG02")
    data["OTRDG02_on"] = emittance_result_OTRDG02_on

    # remove OTRDG02 and insert OTRDG04
    env.remove_screen("OTRDG02")
    env.insert_screen("OTRDG04")

    # turn off TCAV
    env.tcav.amp_set = 0.0

    # run automatic emittance measurement with TCAV off
    emittance_result_OTRDG04_off, _, X = run_automatic_emittance(env, "OTRDG04")
    data["OTRDG04_off"] = emittance_result_OTRDG04_off

    # turn on TCAV
    env.tcav.amp_set = env.tcav_on_amp

    # run automatic emittance measurement with TCAV on
    emittance_result_OTRDG04_on, _, X = run_automatic_emittance(env, "OTRDG04")
    data["OTRDG04_on"] = emittance_result_OTRDG04_on

    # set the tcav amp back to 0.0
    env.tcav.amp_set = 0.0

    # save the results
    saver = H5Saver(save_filename)
    saver.save(data)

    return data

    

