import utils_meas as utils


prj = "obs_amc"
run = "rmte_vessel_0"

n_test = "measurements/vessel/0"

utils.single_observer(prj, run, n_test)



# Per test su simulazioni:
# n_sim = "simulations/BH_8Obs"

# utils.single_observer(prj, run, n_sim)