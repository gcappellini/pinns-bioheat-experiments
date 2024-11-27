import utils as uu
import os
import numpy as np
import common as co
import wandb
import plots as pp
from omegaconf import OmegaConf
from hydra import compose

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)


def main():
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    config = compose(config_name='config_run')
    out_dir = config.output_dir
    dict_exp = config.experiment

    output_dir_gt, _ = co.set_run(out_dir, config, "ground_truth")

    if dict_exp["ground_truth"]:

        # Step 1. execute
        uu.run_matlab_ground_truth()
        pp.plot_matlab_ground_truth(output_dir_gt)

    system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config, path=output_dir_gt)

    if dict_exp["simulation_system"]:

        # Step 0. prepare for simulation_system
        output_dir_system, cfg_system = co.set_run(out_dir, config, "simulation_system")

        # Step 1. execute
        pinns_sys = uu.train_model(cfg_system)
        
        # Step 2. load data
        system = uu.get_pred(pinns_sys, system_gt["grid"], output_dir_system, "system")

        # Step 3. plot data
        uu.check_pinns_and_wandb_upload(system_gt=system_gt, system=system, conf=cfg_system, output_dir=output_dir_system)

    if dict_exp["simulation_mm_obs"]:
    
        # Step 0. prepare for simulation_obs
        output_dir_inverse, config_inverse = co.set_run(out_dir, config, "simulation_mm_obs")
    
        # Step 1. execute
        multi_obs = uu.execute(config_inverse, "simulation_mm_obs")

        # Step 2. load data
        x_obs = uu.gen_obsdata(config_inverse, path=output_dir_gt)
        observers, mm_obs = uu.get_observers_preds(multi_obs, x_obs, output_dir_inverse, config_inverse)

        # Step 3. plot data
        uu.check_pinns_and_wandb_upload(mm_obs_gt=mm_obs_gt, mm_obs=mm_obs, system_gt=system_gt, conf=config_inverse, output_dir=output_dir_inverse, observers=observers, observers_gt=observers_gt)


if __name__ == "__main__":

    main()