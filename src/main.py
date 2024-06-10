import utils

prj = "simulations"
run = "try_2json"
_, output_dir, model_dir, figures_dir = utils.set_name(prj, run)



# Create NBHO with some config.json
config = utils.read_config(run)
config["iterations"] = 30000
utils.write_config(config, run)

n_test = 1
utils.single_observer(prj, run, n_test)

# # Choose dataset to test NBHO (next: generate simulation data with dolfinx)
# n_test = 1
# utils.get_properties(n_test)

# # Next: Either train or restore a model, if the config.json is exactly the same. Names of the runs can be different.
# m = utils.train_model(run)

# # Plots and metrics
# metrics = utils.plot_and_metrics(m, n_test)




