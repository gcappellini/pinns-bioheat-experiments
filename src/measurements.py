import utils as uu
import os
import numpy as np
from scipy import integrate
import common as co
import wandb
import plots as pp
import argparse
from import_vessel_data import load_measurements, extract_entries
from omegaconf import OmegaConf
from datetime import datetime

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)


def scale_predictions(multi_obs, x_obs, prj_figs, lam):
    """
    Generates and scales predictions from the multi-observer model.
    """
    positions = uu.get_tc_positions()
    mm_obs_pred = uu.mm_predict(multi_obs, lam, x_obs, prj_figs)
    preds = np.vstack((x_obs[:, 0], x_obs[:, -1], mm_obs_pred)).T
    
    # Extract predictions based on positions
    y2_pred_sc = preds[preds[:, 0] == positions[0]][:, 2]
    gt2_pred_sc = preds[preds[:, 0] == positions[1]][:, 2]
    gt1_pred_sc = preds[preds[:, 0] == positions[2]][:, 2]
    y1_pred_sc = preds[preds[:, 0] == positions[3]][:, 2]
    
    # Rescale predictions
    y2_pred = uu.rescale_t(y2_pred_sc)
    gt2_pred = uu.rescale_t(gt2_pred_sc)
    gt1_pred = uu.rescale_t(gt1_pred_sc)
    y1_pred = uu.rescale_t(y1_pred_sc)
    
    return y1_pred, gt1_pred, gt2_pred, y2_pred


def main():
    # Parse the config path argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # Load the configuration from the passed file
    config = OmegaConf.load(args.config_path)

    # Now you can access your config values
    experiment_type = config.experiment.type
    print(f"Running measurement with experiment type: {experiment_type}")

    n_obs = config.model_parameters.n_obs
    prj_name = config.experiment.name
    output_dir = co.set_prj(prj_name)

    # Generate and check observers if needed
    multi_obs = uu.mm_observer(n_obs, config)

    # Import data
    a = uu.import_testdata()
    X = a[:, 0:2]
    meas = a[:, 2:3]
    x_obs = uu.import_obsdata()

    lam = config.model_parameters.lam

    y1_pred, gt1_pred, gt2_pred, y2_pred = scale_predictions(multi_obs, x_obs, run_figs, lam)

    # Load and plot timeseries data
    file_path = f"{src_dir}/data/vessel/20240522_1.txt"
    timeseries_data = load_measurements(file_path)
    df = extract_entries(timeseries_data, 83*60, 4*60*60)


    # Optionally check observers and upload to wandb
    uu.check_observers_and_wandb_upload(multi_obs, x_obs, X, meas, config, output_dir, config)

    run_figs = co.set_run(f"mm_obs")
    config.model_properties.W = None
    OmegaConf.save(config, f"{run_figs}/config.yaml")
    
    # Solve IVP and plot weights
    uu.solve_ivp_and_plot(multi_obs, run_figs, n_obs, x_obs, X, meas, lam)

    # Plot time series with predictions
    pp.plot_timeseries_with_predictions(df, y1_pred, gt1_pred, gt2_pred, y2_pred, run_figs)


if __name__ == "__main__":

    main()