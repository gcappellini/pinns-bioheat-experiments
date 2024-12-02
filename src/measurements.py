import os
import numpy as np
from omegaconf import OmegaConf
from hydra import compose
import utils as uu
import common as co
from plots import plot_timeseries_with_predictions

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)



def main():
    config = compose(config_name="config_run")
    out_dir = config.output_dir
    dict_exp = config.experiment
    string = dict_exp["measurements"]

    output_dir_meas, config_meas = co.set_run(out_dir, config, "simulation_mm_obs")
    multi_obs = uu.execute(config_meas, "simulation_mm_obs")

    # Import data
    tot_true = uu.import_testdata(string)
    tot_true = tot_true[:, :-1]
    x_obs = uu.import_obsdata(string)

    # Optionally check observers and upload to wandb
    tot_pred = uu.get_observers_preds(multi_obs, x_obs, output_dir_meas, config_meas)
    uu.check_and_wandb_upload(
        system_gt=system_gt,
        system=system,
        conf=cfg_system,
        output_dir=output_dir_system
    )


    y1_pred, gt1_pred, gt2_pred, y2_pred = uu.point_predictions(tot_pred)
    df = uu.load_from_pickle(f"{src_dir}/data/vessel/{string}.pkl")
    plot_timeseries_with_predictions(df, y1_pred, gt1_pred, gt2_pred, y2_pred, run_figs)


if __name__ == "__main__":
    main()