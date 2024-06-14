import utils

# prj = "simulations"
# n = [1, 2, 3, 4]
n_test = 0
n_obs = 9
var = 0.2


# for n_test in n:
# run = f"change_ic_test{n_test}"
# _, run_dir, model_dir, figures_dir = utils.set_name(prj, run)

# Create NBHO with some config.json
# config = utils.read_config(run)
# config["output_injection_gain"] = 200
# utils.write_config(config, run)

# utils.single_observer(prj, run, n_test)
utils.mm_observer_k(n_test, n_obs, var)





