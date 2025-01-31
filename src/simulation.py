import os, logging
import numpy as np
from omegaconf import OmegaConf
from hydra import compose
import utils as uu
import common as co
import plots as pp
from common import setup_logging

# Directories Setup
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)

logger = setup_logging()

def run_ground_truth(config, out_dir):
    """Run MATLAB ground truth simulation, load data, and plot results."""
    logger.info("Running MATLAB ground truth simulation.")
    label = "ground_truth"
    dict_exp = config.experiment
    output_dir_gt, config_matlab = co.set_run(out_dir, config, label)
    uu.run_matlab_ground_truth()
    system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config_matlab, path=out_dir)
    system_meas, _ = uu.import_testdata(config)

    # uu.compute_metrics([system_gt, *observers_gt, mm_obs_gt], config, out_dir)

    if config.experiment.plot:
        if config.model_parameters.n_obs == 0:
            pp.plot_multiple_series([system_gt, system_meas], out_dir, label)

        elif config.model_parameters.n_obs == 1:
            pp.plot_multiple_series([system_gt, *observers_gt], out_dir, label)
            pp.plot_l2(system_gt, [*observers_gt], out_dir, label)
            pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], mm_obs_gt["theta"], out_dir, label)
            pp.plot_obs_err([*observers_gt], out_dir, label)

        elif config.plot.show_obs:
            pp.plot_multiple_series([system_gt, *observers_gt, mm_obs_gt], out_dir, label)
            pp.plot_l2(system_gt, [*observers_gt, mm_obs_gt], out_dir, label)
            pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], mm_obs_gt["theta"], out_dir, label)
            pp.plot_obs_err([*observers_gt, mm_obs_gt], out_dir, label)

            if 1 < config.model_parameters.n_obs <= 8:
                pp.plot_weights([*observers_gt], out_dir, label)
        
        else:
            pp.plot_multiple_series([system_gt, mm_obs_gt], out_dir, label)
            pp.plot_l2(system_gt, [mm_obs_gt], out_dir, label)
            pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], mm_obs_gt["theta"], out_dir, label)
            pp.plot_obs_err([mm_obs_gt], out_dir, label)

            if dict_exp["run"].startswith("meas"):
                system_meas, _ = uu.import_testdata(config)
                pp.plot_timeseries_with_predictions(system_meas, system_gt, config, out_dir)     
        
    
    return output_dir_gt, system_gt, observers_gt, mm_obs_gt


def run_simulation_system(config, out_dir, system_gt):
    """Run simulation for the system and plot results."""
    logger.info("Running simulation for the system.")
    label = "simulation_system"
    output_dir_system, cfg_system = co.set_run(out_dir, config, label)
    pinns_sys = uu.train_model(cfg_system)
    system = uu.get_pred(pinns_sys, system_gt["grid"], output_dir_system, "system")

    uu.compute_metrics([system_gt, system], config, out_dir)

    if config.experiment.plot:
        pp.plot_multiple_series([system_gt, system], out_dir, label)
        pp.plot_l2(system_gt, [system], out_dir, label)
        pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], system["theta"], out_dir, label)
        pp.plot_obs_err([system], out_dir, label)




def run_simulation_mm_obs(config, out_dir, system_gt, mm_obs_gt, observers_gt, gt_path=None):
    """Run multi-observer simulation, load data, and plot results."""
    logger.info("Running simulation for multi-observer.")
    label = "simulation_mm_obs"
    output_dir_inverse, config_inverse = co.set_run(out_dir, config, label)
    multi_obs = uu.execute(config_inverse, label)
    x_obs = uu.gen_obsdata(config_inverse, system_gt)
    observers, mm_obs = uu.get_observers_preds(system_gt, multi_obs, x_obs, out_dir, config_inverse, label)

    uu.compute_metrics([system_gt, *observers, mm_obs], config, out_dir)

    if config.experiment.plot:

        if config.plot.show_obs:
            pp.plot_multiple_series([system_gt, *observers, mm_obs_gt, mm_obs], out_dir, label)
            pp.plot_l2(system_gt, [*observers, mm_obs_gt, mm_obs], out_dir, label)
            pp.plot_obs_err([*observers, mm_obs_gt, mm_obs], out_dir, label)
            if 1 < config.model_parameters.n_obs <= 8:
                pp.plot_weights([*observers], out_dir, label)
        else:
            pp.plot_multiple_series([system_gt, mm_obs_gt, mm_obs], out_dir, label)
            pp.plot_l2(system_gt, [mm_obs_gt, mm_obs], out_dir, label)
            pp.plot_obs_err([mm_obs_gt, mm_obs], out_dir, label)

        pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], mm_obs["theta"], out_dir, label)



def run_measurement_mm_obs(config, out_dir):
    """Run multi-observer simulation, load data, and plot results."""
    label = config.experiment.run
    logger.info(f"Running measurement {label} for multi-observer")
    output_dir_meas, config_meas = co.set_run(out_dir, config, label)
    multi_obs = uu.execute(config_meas, label)
    system_meas, _ = uu.import_testdata(config_meas)
    x_obs = uu.import_obsdata(config_meas)
    observers, mm_obs = uu.get_observers_preds(system_meas, multi_obs, x_obs, out_dir, config_meas, label)

    uu.compute_metrics([system_meas, *observers, mm_obs], config, out_dir)

    if config.experiment.plot:

        if config.plot.show_obs:
            pp.plot_multiple_series([system_meas, *observers, mm_obs], out_dir, label)
            pp.plot_l2(system_meas, [*observers, mm_obs], out_dir, label)
            # pp.plot_validation_3d(system_meas["grid"], system_meas["theta"], mm_obs["theta"], out_dir, label)
            pp.plot_obs_err([*observers, mm_obs], out_dir, label)
            if 1 < config.model_parameters.n_obs <= 8: 
                pp.plot_weights([*observers], out_dir, label)
        else:
            pp.plot_multiple_series([system_meas, mm_obs], out_dir, label)
            pp.plot_l2(system_meas, [mm_obs], out_dir, label)
            # pp.plot_validation_3d(system_meas["grid"], system_meas["theta"], mm_obs["theta"], out_dir, label)
            pp.plot_obs_err([mm_obs], out_dir, label)

        pp.plot_timeseries_with_predictions(system_meas, mm_obs, config, out_dir)


def main():
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    config = compose(config_name="config_run")
    run_out_dir = config.output_dir
    dict_exp = config.experiment
    n_ins = config.model_properties.n_ins

    gt_path=f"{tests_dir}/cooling_ground_truth"

    if dict_exp["simulation"]:
        if dict_exp["ground_truth"]:
            output_dir_gt, system_gt, observers_gt, mm_obs_gt = run_ground_truth(config, run_out_dir)
        else:
            system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config, path=gt_path)
        # Simulation System
        if n_ins==2:
            run_simulation_system(config, run_out_dir, system_gt, gt_path)

        # Simulation Multi-Observer
        else:
            run_simulation_mm_obs(config, run_out_dir, system_gt, mm_obs_gt, observers_gt, gt_path)
    
    elif dict_exp["run"].startswith("meas"):
        if dict_exp["ground_truth"]:
            output_dir_gt, system_gt, observers_gt, mm_obs_gt = run_ground_truth(config, run_out_dir)
            
        run_measurement_mm_obs(config, run_out_dir)


if __name__ == "__main__":
    main()