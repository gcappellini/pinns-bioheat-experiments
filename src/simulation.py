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


def main():
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    config = OmegaConf.load(f"{src_dir}/config.yaml")
    X, y_sys, y_obs, _ = uu.gen_testdata(config)
    x_obs = uu.gen_obsdata(config)
    tot_true = np.hstack((X, y_sys, y_obs))

    # Load the configuration from the passed file
    config = OmegaConf.load(f"{src_dir}/config.yaml")
    
    prj_name = config.experiment.name
    check_system = config.experiment.check_system
    name_str = f"{prj_name[0]}_{prj_name[1]}"

    if check_system:
        output_dir = co.set_prj(f"{name_str}/simulation_system")
        config.model_properties.W = config.model_parameters.W4
        config.model_properties.direct = True
        OmegaConf.save(config,f"{output_dir}/config.yaml")

        pinns_sys = uu.train_model(output_dir, system=True)

        y_sys_pinns = uu.get_system_pred(pinns_sys, X, output_dir)
        uu.check_system_and_wandb_upload(tot_true[:, :-1], y_sys_pinns, config, output_dir)

    n_obs = config.model_parameters.n_obs
    config.model_properties.direct = False
    output_dir = co.set_prj(f"{name_str}/simulation_{n_obs}obs")
    OmegaConf.save(config,f"{output_dir}/config.yaml")

    # Generate and check observers if needed
    multi_obs = uu.mm_observer(config)

    tot_pred = uu.get_observers_preds(multi_obs, x_obs, output_dir, config)
    metrics = uu.check_observers_and_wandb_upload(tot_true, tot_pred, config, output_dir)
    print(metrics)
    if n_obs==1:
        pp.plot_validation_3d(tot_true[:, 0:2], tot_true[:, -1], tot_pred[:, -1], output_dir)
        instants = [0, 0.25, 0.5, 0.75]
        for t in instants:
            pp.plot_tx(t, tot_true, tot_pred, 0, output_dir, MultiObs=False)
    if n_obs>1:
        run_figs = co.set_run(f"mm_obs")
        pp.plot_mm_obs(multi_obs, tot_true, tot_pred, config, run_figs)


if __name__ == "__main__":

    main()