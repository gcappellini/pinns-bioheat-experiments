import utils_cdc as utils

n_test="0-cdc"
prj = "single_obs"
utils.set_prj(prj)

run = f"mac_{n_test}_new"
utils.set_run(run)


utils.single_observer(prj, run, n_test)


# utils.mm_observer(n_test, 8, 0.8)

