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

    uu.run_matlab_ground_truth(src_dir, prj_figs, conf, True)


if __name__ == "__main__":

    # Load the configuration from the passed file
    config = OmegaConf.load(f"{src_dir}/config.yaml")

    prj_name = config.experiment.name
    name_str = f"{prj_name[0]}_{prj_name[1]}"
    
    n_obs = config.model_parameters.n_obs

    output_dir = co.set_prj(name_str)

    cfg = uu.configure_meas_settings(config, config.experiment.name)

    cfg_matlab = OmegaConf.create({
    "model_properties": cfg.model_properties,
    "model_parameters": cfg.model_parameters
})

    OmegaConf.save(cfg_matlab,f"{output_dir}/config_matlab.yaml")
    OmegaConf.save(cfg,f"{output_dir}/config.yaml")
    OmegaConf.save(cfg_matlab,f"{src_dir}/config_matlab.yaml")
    main(output_dir, cfg_matlab)