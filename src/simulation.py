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


def main(prj_figs, conf):
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    n_obs = conf.model_parameters.n_obs

    # Generate and check observers if needed
    multi_obs = uu.mm_observer(conf)
    X, y_sys, _, _ = uu.gen_testdata(n_obs)
    x_obs = uu.gen_obsdata(n_obs)
    uu.check_observers_and_wandb_upload(multi_obs, x_obs, X, y_sys, conf, prj_figs)
    uu.check_mm_obs(multi_obs, x_obs, X, y_sys, conf)


if __name__ == "__main__":

    # Load the configuration from the passed file
    config = OmegaConf.load(f"{src_dir}/config.yaml")
    
    prj_name = config.experiment.name
    name_str = f"{prj_name[0]}_{prj_name[1]}"

    n_obs = config.model_parameters.n_obs
    
    output_dir = co.set_prj(f"{name_str}/simulation_{n_obs}obs")
    cfg = uu.configure_meas_settings(config, config.experiment.name)

    OmegaConf.save(cfg,f"{output_dir}/config.yaml")
    main(output_dir, config)