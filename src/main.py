import utils

prj = "simulations"
run = "corrected_bc"
_, output_dir, model_dir, figures_dir = utils.set_name(prj, run)



# Create NBHO with some config.json
config = utils.read_config(run)
# config["output_injection_gain"] = 200
utils.write_config(config, run)

n_test = 1
utils.single_observer(prj, run, n_test)





