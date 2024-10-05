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

    # Optionally run MATLAB ground truth
    uu.run_matlab_ground_truth(src_dir, prj_figs, conf, True)


if __name__ == "__main__":

    # Load the configuration from the passed file
    config = OmegaConf.load(f"{src_dir}/config.yaml")

    # Now you can access your config values
    experiment_type = config.experiment.type
    print(f"Running experiment type: {experiment_type}")
    prj_name = config.experiment.name
    
    output_dir = co.set_prj(prj_name)

    OmegaConf.save(config,f"{output_dir}/config.yaml")
    main(output_dir, config, run_matlab=config.experiment.run_matlab, run_wandb=config.experiment.run_wandb)