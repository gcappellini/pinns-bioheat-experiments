import utils

prj = "simulations"
n = [1, 2, 3, 4]

for n_test in n:
    run = f"single_obs_default_{n_test}"
    _, output_dir, model_dir, figures_dir = utils.set_name(prj, run)

    # Create NBHO with some config.json
    config = utils.read_config(run)
    # config["output_injection_gain"] = 200
    utils.write_config(config, run)

    utils.single_observer(prj, run, n_test)





