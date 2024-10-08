import utils as uu
import os
import numpy as np
import common as co
import wandb
from omegaconf import OmegaConf
import plots as pp

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)



def main(config, output_dir):

    # Generate and check observers if needed
    multi_obs = uu.mm_observer(config)
    name = config.experiment.name
    string = f"{name[0]}_{name[1]}"

    # Import data
    a = uu.import_testdata(string)
    X = a[:, 0:2]
    meas = a[:, 2:3]
    x_obs = uu.import_obsdata(string)

    lam = config.model_parameters.lam

    # Optionally check observers and upload to wandb
    uu.check_observers_and_wandb_upload(multi_obs, x_obs, X, meas, config, output_dir, comparison_3d=False)
    uu.check_mm_obs(multi_obs, x_obs, X, meas, config, comparison_3d=False)

    run_figs = co.set_run(f"mm_obs")
    y1_pred, gt1_pred, gt2_pred, y2_pred = uu.point_predictions(multi_obs, x_obs, run_figs, lam)

    df = uu.load_from_pickle(f"{src_dir}/data/vessel/{string}.pkl")
    pp.plot_timeseries_with_predictions(df, y1_pred, gt1_pred, gt2_pred, y2_pred, run_figs)


if __name__ == "__main__":

    # Load the configuration from the passed file
    config = OmegaConf.load(f"{src_dir}/config.yaml")
    
    prj_name = config.experiment.name
    name_str = f"{prj_name[0]}_{prj_name[1]}"
    
    n_obs = config.model_parameters.n_obs

    output_dir = co.set_prj(f"{name_str}/measurements_{n_obs}obs")
    cfg = uu.configure_meas_settings(config, config.experiment.name)

    OmegaConf.save(cfg,f"{output_dir}/config.yaml")

    main(config, output_dir)