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
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)


def main():
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    config = OmegaConf.load(f"{conf_dir}/config_run.yaml")
    out_dir = config.output_dir

    if config.experiment.run_matlab:
        output_dir = co.set_run(out_dir, f"ground_truth")
        uu.run_matlab_ground_truth(output_dir)

    if config.experiment.check_system:
        config.model_properties.W = config.model_parameters.W_sys
        config.model_properties.direct = True
        
        output_dir = co.set_run(out_dir, "simulation_system")

        pinns_sys = uu.train_model(output_dir, system=True)
        
        tot_true = uu.gen_testdata(config)
        y_sys_pinns = uu.get_system_pred(pinns_sys, tot_true[:, 0:2], output_dir)
        uu.check_system_and_wandb_upload(tot_true[:, :3], y_sys_pinns, config, output_dir)

    # x_obs = uu.gen_obsdata(config)
    # n_obs = config.model_parameters.n_obs
    # config.model_properties.direct = False
    # output_dir = co.set_run(out_dir, f"simulation_{n_obs}obs")

    # # Generate and check observers if needed
    # multi_obs = uu.mm_observer(config)

    # tot_pred = uu.get_observers_preds(multi_obs, x_obs, output_dir, config)
    # uu.check_observers_and_wandb_upload(tot_true, tot_pred, config, output_dir)

    # if n_obs>1:
    #     run_figs = co.set_run(f"mm_obs")
    #     pp.plot_mm_obs(multi_obs, tot_true, tot_pred, run_figs)


if __name__ == "__main__":

    main()