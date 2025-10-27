import logging 

logger = logging.getLogger("auto_6d")

from ml_tto.diag0.auto_emittance import run_automatic_emittance
from lcls_tools.common.data.saver import H5Saver
import time
import pandas as pd
import yaml

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
    saver = H5Saver()

    # turn off TCAV
    env.tcav.amplitude = 0.0
    time.sleep(5.0)

    data = {}

    # run automatic emittance measurement with TCAV off
    logger.info("running OTRDG02 quad scan tcav off")
    emittance_result_OTRDG02_off, _, X = run_automatic_emittance(env, "OTRDG02")
    data["OTRDG02_off"] = emittance_result_OTRDG02_off.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }
    # save the results
    tracking_data = X.data
    saver.dump(data, save_filename)

    # turn on TCAV
    env.tcav.amplitude = env.tcav_on_amp
    time.sleep(5.0)

    # run automatic emittance measurement with TCAV on
    logger.info("running OTRDG02 quad scan tcav on")
    emittance_result_OTRDG02_on, _, X = run_automatic_emittance(env, "OTRDG02")
    data["OTRDG02_on"] = emittance_result_OTRDG02_on.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }
    # save the results
    tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)
    saver.dump(data, save_filename)

    # remove OTRDG02 and insert OTRDG04
    env.set_screen("OTRDG04")

    # turn off TCAV
    env.tcav.amplitude = 0.0
    time.sleep(5.0)

    # run automatic emittance measurement with TCAV off
    logger.info("running OTRDG04 quad scan tcav off")
    emittance_result_OTRDG04_off, _, X = run_automatic_emittance(env, "OTRDG04")
    data["OTRDG04_off"] = emittance_result_OTRDG04_off.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }
    # save the results
    tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)
    saver.dump(data, save_filename)

    # turn on TCAV
    env.tcav.amplitude = env.tcav_on_amp
    time.sleep(5.0)

    # run automatic emittance measurement with TCAV on
    logger.info("running OTRDG04 quad scan tcav on")
    emittance_result_OTRDG04_on, _, X = run_automatic_emittance(env, "OTRDG04")
    data["OTRDG04_on"] = emittance_result_OTRDG04_on.model_dump() | {
        "environment_variables": env.get_variables(env.variables.keys())
    }

    # set the tcav amp back to 0.0
    env.tcav.amplitude = 0.0

    # save the results
    tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)
    saver.dump(data, save_filename)

    return data, tracking_data
