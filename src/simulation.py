import os
import numpy as np
from omegaconf import OmegaConf
from hydra import compose
import utils as uu
import common as co

# Directories Setup
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)


def run_ground_truth(config, out_dir):
    """Run MATLAB ground truth simulation, load data, and plot results."""
    label = "ground_truth"
    output_dir_gt, config_matlab = co.set_run(out_dir, config, label)
    uu.run_matlab_ground_truth()
    system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config_matlab, path=output_dir_gt)
    if config.experiment.plot:
        uu.check_and_wandb_upload(
            label=label,
            mm_obs_gt=mm_obs_gt,
            system_gt=system_gt,
            conf=config,
            output_dir=output_dir_gt,
            observers_gt=observers_gt
        )
    # system_meas, _ = uu.import_testdata(config)
    # uu.check_measurements(system_meas, system_gt, output_dir_gt, config)
    # extracted = uu.extract_matching([system_gt, system_meas])
    # metric = uu.compute_metrics(extracted, [system_gt, system_meas], config, out_dir)
    # score = metric["system_meas_L2RE"]
    
    return output_dir_gt, system_gt, observers_gt, mm_obs_gt


def run_simulation_system(config, out_dir, system_gt):
    """Run simulation for the system and plot results."""
    label = "simulation_system"
    output_dir_system, cfg_system = co.set_run(out_dir, config, label)
    pinns_sys = uu.train_model(cfg_system)
    system = uu.get_pred(pinns_sys, system_gt["grid"], output_dir_system, "system")
    if config.experiment.plot:
        uu.check_and_wandb_upload(
            label=label,
            system_gt=system_gt,
            system=system,
            conf=cfg_system,
            output_dir=output_dir_system
        )


def run_simulation_mm_obs(config, out_dir, output_dir_gt, system_gt, mm_obs_gt, observers_gt):
    """Run multi-observer simulation, load data, and plot results."""
    label = "simulation_mm_obs"
    output_dir_inverse, config_inverse = co.set_run(out_dir, config, label)
    multi_obs = uu.execute(config_inverse, label)
    x_obs = uu.gen_obsdata(config_inverse, path=output_dir_gt)
    observers, mm_obs = uu.get_observers_preds(multi_obs, x_obs, output_dir_inverse, config_inverse, label)
    if config.experiment.plot:
        uu.check_and_wandb_upload(
            label=label,
            mm_obs_gt=mm_obs_gt,
            mm_obs=mm_obs,
            system_gt=system_gt,
            conf=config_inverse,
            output_dir=output_dir_inverse,
            observers=observers,
            observers_gt=observers_gt
        )


def run_measurement_mm_obs(config, out_dir):
    """Run multi-observer simulation, load data, and plot results."""
    label = config.experiment.run
    output_dir_meas, config_meas = co.set_run(out_dir, config, label)
    multi_obs = uu.execute(config_meas, label)
    system_meas, _ = uu.import_testdata(config_meas)
    x_obs = uu.import_obsdata(config_meas)
    observers, mm_obs = uu.get_observers_preds(multi_obs, x_obs, output_dir_meas, config_meas, label)
    if config.experiment.plot:
        uu.check_and_wandb_upload(
            label=label,
            mm_obs=mm_obs,
            system_meas=system_meas,
            conf=config_meas,
            output_dir=output_dir_meas,
            observers=observers
        )
        uu.check_measurements(system_meas, mm_obs, output_dir_meas, config_meas)


def load_ground_truth(config, out_dir):
    output_dir_gt, config_matlab = co.set_run(out_dir, config, "ground_truth")
    system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config_matlab, path=output_dir_gt)
    return output_dir_gt, system_gt, observers_gt, mm_obs_gt


def main():
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    config = compose(config_name="config_run")
    run_out_dir = config.output_dir
    dict_exp = config.experiment
    n_ins = config.model_properties.n_ins

    # Ground Truth Simulation
    if dict_exp["ground_truth"]:
        output_dir_gt, system_gt, observers_gt, mm_obs_gt = run_ground_truth(config, run_out_dir)
    # else:
    #     output_dir_gt, system_gt, observers_gt, mm_obs_gt = load_ground_truth(config, f"{tests_dir}/cooling_simulation_{n_obs}obs")

    if dict_exp["run"]=="simulation":
        # Simulation System
        if n_ins==2:
            run_simulation_system(config, run_out_dir, system_gt)

        # Simulation Multi-Observer
        else:
            run_simulation_mm_obs(config, run_out_dir, output_dir_gt, system_gt, mm_obs_gt, observers_gt)
    
    elif dict_exp["run"].startswith("meas"):
        run_measurement_mm_obs(config, run_out_dir)


if __name__ == "__main__":
    main()