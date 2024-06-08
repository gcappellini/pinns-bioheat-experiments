import utils

prj = None
run = None
_, output_dir, model_dir, figures_dir = utils.set_name(prj, run)


# Create NBHO with some config.json
config = utils.read_config(run)
config["iterations"] = 2
utils.write_config(config, run)
a = utils.create_nbho(config)

# Create dataset to test NBHO (mathematica, next: dolfinx)
e, f = utils.gen_testdata(1)
g = utils.gen_obsdata(1)

# Either train or restore a model, if the config.json is exactly the same. Names of the runs can be different.



# Plots and metrics


