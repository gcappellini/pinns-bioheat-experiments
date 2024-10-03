import utils as uu
import os
import numpy as np
import common as co
import wandb
import plots as pp
from omegaconf import OmegaConf

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)


def main(prj_figs, conf, run_matlab=False, run_wandb=False):
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    n_obs = conf.model_parameters.n_obs
    # Optionally run MATLAB ground truth
    # uu.run_matlab_ground_truth(src_dir, prj_figs, conf, run_matlab)

    # # Generate and check observers if needed
    multi_obs = uu.mm_observer(conf)
    # X, y_sys, _, _ = uu.gen_testdata(n_obs)
    # x_obs = uu.gen_obsdata(n_obs)
    # uu.check_observers_and_wandb_upload(multi_obs, x_obs, X, y_sys, conf, prj_figs)
    # uu.check_mm_obs(multi_obs, x_obs, X, y_sys, conf, prj_figs)


if __name__ == "__main__":

    # Load the configuration from the passed file
    config = OmegaConf.load(f"{src_dir}/config.yaml")

    # Now you can access your config values
    experiment_type = config.experiment.type
    print(f"Running measurement with experiment type: {experiment_type}")
    prj_name = config.experiment.name
    
    output_dir = co.set_prj(prj_name)

    OmegaConf.save(config,f"{output_dir}/config.yaml")
    main(output_dir, config, run_matlab=config.experiment.run_matlab, run_wandb=config.experiment.run_wandb)