import logging
import yaml
import time
import traceback
import pandas as pd
from tenacity import RetryError

from ml_tto.diag0.auto_alignment import run_automatic_alignment
from ml_tto.diag0.auto_tcav_phasing import run_automatic_tcav_phasing
from ml_tto.diag0.auto_6d import run_automatic_6d_measurement
from ml_tto.diag0.auto_focusing import run_auto_focusing
from ml_tto.diag0.utils import process_data
from xopt.errors import XoptError


logger = logging.getLogger("auto_diag0")


def run_automatic_diag0(env, save_location="data/"):

    ## reset the tcav
    env.tcav.amplitude = 0.0
    env.tcav.phase = 67.0

    reset_vals = env.get_variables(list(env.variables.keys()))

    # run alignment
    try:
        while(True):
            try:
                ts = time.time()
                start = time.time()
        
                logger.info(f"starting diag0 measurement at {int(ts)}")
                env.otrdg02_inserted = False
                env.otrdg04_inserted = True
            
                logger.info("starting alignment")
                X = run_automatic_alignment(env, n_steps=10, to_screen_name="OTRDG04",target_value=1.5)
                X.dump(f"data/alignment_{int(ts)}.yaml")
                alignment_time = time.time()
                logger.info(f"alignment time: {alignment_time - start}")
        
                tracking_data = X.data
                tracking_data["process"] = "alignment"
            
                # update reset vals
                reset_vals = env.get_variables(list(env.variables.keys()))
        
                # run focusing
                logger.info("starting otrdg02 focusing")
                quads = ["QUAD:DIAG0:360:BCTRL","QUAD:DIAG0:370:BCTRL","QUAD:DIAG0:390:BCTRL"]
                X = run_auto_focusing(env, "OTRDG02", quads, target_value = 140.0, n_steps=10)
                X.data["process"] = "otrdg02_focusing"
                tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)  
        
                logger.info("starting otrdg04 focusing")
                X = run_auto_focusing(
                    env,
                    "OTRDG04",
                    [
                        "QUAD:DIAG0:455:BCTRL",
                        "QUAD:DIAG0:470:BCTRL",
                    ], 
                    target_value=160000.0,
                    objective="prod_size", 
                    n_steps=10
                )        
                X.data["process"] = "otrdg04_focusing"
                tracking_data = pd.concat([tracking_data, X.data], ignore_index=True) 
                
                # update reset vals
                reset_vals = env.get_variables(list(env.variables.keys()))
        
                logger.info("starting alignment")
                env.otrdg02_inserted = False
                X = run_automatic_alignment(env, n_steps=10, to_screen_name="OTRDG04",target_value=1.5)
                X.dump(f"data/alignment_{int(ts)}.yaml")
                alignment_time = time.time()
                logger.info(f"alignment time: {alignment_time - start}")
        
                X.data["process"] = "alignment"
                tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)  
        
                # update reset vals
                reset_vals = env.get_variables(list(env.variables.keys()))
                
                # run tcav phasing
                logger.info("phasing tcav")
                env.otrdg02_inserted = False
                env.tcav.amplitude = env.tcav_on_amp
                X = run_automatic_tcav_phasing(env)
                phasing_time = time.time()
                logger.info(f"tcav phasing time: {phasing_time - alignment_time}")
        
                X.data["process"] = "tcav_phasing"
                tracking_data = pd.concat([tracking_data, X.data], ignore_index=True)
            
                # update reset vals
                reset_vals = env.get_variables(list(env.variables.keys()))
                
                # run 6d measurement
                logger.info("starting 6d measurement")
                raw_data_file = f"data/6d_data_{int(ts)}.h5"
                _, track_data = run_automatic_6d_measurement(env, raw_data_file)
                gpsr_time = time.time()
                logger.info(f"gpsr time: {gpsr_time - phasing_time}")
                logger.info(f"total time: {time.time() - start}")
        
                track_data["process"] = "gpsr"
                tracking_data = pd.concat([tracking_data, track_data], ignore_index=True)
        
                # save top level tracking data
                with open(f"data/tracking_{int(ts)}.yaml", 'w') as file:
                    yaml.dump(
                        tracking_data.to_dict(), 
                        file, 
                        default_flow_style=False
                    )

                # process data
                process_data(raw_data_file, env.save_directory + "processed_data/")
        
                time.sleep(2)
        
                #try:
                #    process_data("data/" + raw_data_file, "processed_data/", visualize=False, minimum_transmission=0.95)
                #except RuntimeError:
                #    warnings.warn(f"unable to properly process {raw_data_file}, skipping")
                
                # turn off TCAV
                env.tcav.amplitude = 0.0
            except XoptError as e:
                logger.warning(f"XOPT ERROR {e} RESTARTING")
            except RetryError:
                logger.warning("RUN INTERRUPTED, RESTARTING")
            
    except Exception as e:
        # reset variables if something goes wrong
        print(traceback.format_exc())
        env.tcav.amplitude = 0.0
        env.set_variables(reset_vals)

        raise e
