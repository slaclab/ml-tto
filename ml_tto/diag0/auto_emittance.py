def run_automatic_emittance(env, screen_name):
    """
    Run automatic emittance measurement using the specified environment and screen name.

    Parameters:
        env (Environment): The environment in which the measurement is performed.
        screen_name (str): The name of the screen device to be used for measurements.

    Returns:
        ScreenBeamProfileMeasurementResult: The result of the beam profile measurement.
        fname (str): The filename where the results are saved.
        X: Xopt object from the emittance measurement.
    """
    energy = env.get_variables(["BEND:DIAG0:155:BCTRL"])["BEND:DIAG0:155:BCTRL"] * 1e9

    if screen_name == "OTRDG02":
        env.emittance_config_fname = "/home/physics/badger/resources/dev/plugins/environments/diag0_dev/emittance_measurement_configs/OTRDG02.yaml"
        env.beamsize_cutoff_max = 5.0
        env.min_beamsize_cutoff = 1000
        env.create_beamprofile_measurement()
        env._create_emittance_object()
        env._emittance_measurement_object.reset()
        env._emittance_measurement_object.energy = energy

    elif screen_name == "OTRDG04":
        env.emittance_config_fname = "/home/physics/badger/resources/dev/plugins/environments/diag0_dev/emittance_measurement_configs/OTRDG04.yaml"
        env.beamsize_cutoff_max = 5.0
        env.min_beamsize_cutoff = 1000
        env.create_beamprofile_measurement()
        env._create_emittance_object()
        env._emittance_measurement_object.reset()
        env._emittance_measurement_object.energy = energy

    emittance_result, fname = env.run_emittance_measurement()
    return emittance_result, fname, env._emittance_measurement_object.X
