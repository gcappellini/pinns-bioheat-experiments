import utils_meas as utils


prj = "obs_amc"
run = "try_vessel"

n_test = "measurements/vessel/1"

utils.single_observer(prj, run, n_test)



# Per test su simulazioni:
# n_sim = "simulations/BH_8Obs"

# utils.single_observer(prj, run, n_sim)