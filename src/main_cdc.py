import utils_cdc as utils

n_test="0-cdc"
prj = "single_obs"
utils.set_prj(prj)
run = f"matlab_{n_test}"
utils.set_run(run)
# c = utils.read_config()
# c["output_injection_gain"] = 50000
# utils.write_config(c)
utils.single_observer(prj, run, n_test)


utils.mm_observer(n_test, 8, 0.8)

