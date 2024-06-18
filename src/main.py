import utils

prj = "single_obs"
utils.set_prj(prj)
n_test="0-cdc"

run = f"{n_test}"
utils.set_run(run)
# c = utils.read_config()
# c["output_injection_gain"] = 50000
# utils.write_config(c)
utils.single_observer(prj, run, n_test)


# utils.mm_observer(0, 3, 0.5)
# utils.mm_observer(1, 3, 0.34)

