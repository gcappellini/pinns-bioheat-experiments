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


def main(output_dir, conf):
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    n_obs = conf.model_parameters.n_obs
    # Generate and check observers if needed
    multi_obs = uu.mm_observer(conf)
    X, y_sys, _, _ = uu.gen_testdata(conf)
    x_obs = uu.gen_obsdata(conf)
    tot_true = np.hstack((X, y_sys))
    tot_pred = uu.get_observers_preds(multi_obs, x_obs, output_dir, config)
    uu.check_observers_and_wandb_upload(tot_true, tot_pred, config, output_dir)
    if n_obs>1:
        run_figs = co.set_run(f"mm_obs")
        pp.plot_mm_obs(multi_obs, tot_true, tot_pred, config, run_figs)


if __name__ == "__main__":

    # Load the configuration from the passed file
    config = OmegaConf.load(f"{src_dir}/config.yaml")
    
    prj_name = config.experiment.name
    name_str = f"{prj_name[0]}_{prj_name[1]}"

    n_obs = config.model_parameters.n_obs
    
    output_dir = co.set_prj(f"{name_str}/simulation_{n_obs}obs")

    OmegaConf.save(config,f"{output_dir}/config.yaml")
    main(output_dir, config)