import utils

prj = "single_obs"
utils.set_prj(prj)

for n_test in range(1):
    run = f"default_{n_test}"
    utils.set_run(run)
    utils.single_observer(prj, run, n_test)




