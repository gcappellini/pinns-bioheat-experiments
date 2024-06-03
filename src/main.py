import utils

prj = None
run = None
_, output_dir, model_dir, figures_dir = utils.set_name(prj, run)

# Create NBHO with some config.json
config = utils.read_config(run)
config["iterations"] = 2
utils.write_config(config, run)
a = utils.create_nbho(config) # Check pde formulation!!!
# Either train or restore a model, if the config.json is exactly the same. Names of the runs can be different.

# Create dataset to test NBHO



# Plots and metrics


