from ml_tto.diag0.auto_6d import run_automatic_6d_measurement
from ml_tto.diag0.auto_alignment import run_automatic_alignment
from ml_tto.diag0.auto_tcav_phasing import run_automatic_tcav_phasing
import yaml
import time
import pandas as pd


def run_automatic_diag0(env, save_filename):
    ## reset the tcav
    env.tcav.amplitude = 0.0
    env.tcav.phase = 100.0

    reset_vals = env.get_variables(list(env.variables.keys()))

    # run alignment
    try:
        for i in range(2):
            ts = time.time()
            start = time.time()

            print(f"starting diag0 measurement at {int(ts)}")
            env.otrdg02_inserted = False
            env.otrdg04_inserted = True

            print("starting alignment")
            X = run_automatic_alignment(
                env, n_steps=20, to_screen_name="OTRDG04", target_value=1.5
            )
            X.dump(f"data/alignment_{int(ts)}.yaml")
            alignment_time = time.time()
            print(f"alignment time: {alignment_time - start}")

            tracking_data = X.data
            tracking_data["process"] = "alignment"

            # update reset vals
            reset_vals = env.get_variables(list(env.variables.keys()))

            # run tcav phasing
            print("phasing tcav")
            env.tcav.amplitude = env.tcav_on_amp
            X = run_automatic_tcav_phasing(env)
            phasing_time = time.time()
            print(f"tcav phasing time: {phasing_time - alignment_time}")

            X.data["process"] = "tcav_phasing"
            tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)

            # update reset vals
            reset_vals = env.get_variables(list(env.variables.keys()))

            # run 6d measurement
            print("starting 6d measurement")
            _, track_data = run_automatic_6d_measurement(
                env, f"data/6d_data_{int(ts)}.h5"
            )
            gpsr_time = time.time()
            print(f"gpsr time: {gpsr_time - phasing_time}")
            print(f"total time: {time.time() - start}")

            track_data["process"] = "gpsr"
            tracking_data = pd.concat([tracking_data, track_data], ignore_index=True)

            # save top level tracking data
            with open(f"data/tracking_{int(ts)}.yaml", "w") as file:
                yaml.dump(tracking_data.to_dict(), file, default_flow_style=False)

            # turn off TCAV
            env.tcav.amplitude = 0.0

    except Exception as e:
        # reset variables if something goes wrong
        env.set_variables(reset_vals)
        env.tcav.amplitude = 0.0

        raise e
