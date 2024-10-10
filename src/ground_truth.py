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

    uu.run_matlab_ground_truth(prj_figs, conf, True)


if __name__ == "__main__":

    # Load the configuration from the passed file
    cfg = OmegaConf.load(f"{src_dir}/config.yaml")

    prj_name = cfg.experiment.name
    name_str = f"{prj_name[0]}_{prj_name[1]}/ground_truth"
    
    n_obs = cfg.model_parameters.n_obs

    output_dir = co.set_prj(name_str)

    cfg_matlab = OmegaConf.create({
    "model_properties": cfg.model_properties,
    "model_parameters": cfg.model_parameters,
    "experiment": cfg.experiment.name
    })

    OmegaConf.save(cfg_matlab,f"{output_dir}/config_matlab.yaml")
    OmegaConf.save(cfg,f"{output_dir}/config.yaml")

    main(output_dir, cfg_matlab)