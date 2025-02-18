import os, logging
import numpy as np
from omegaconf import OmegaConf
from hydra import compose
import utils as uu
import common as co
import plots as pp
import deepxde as dde
from common import setup_log
import wandb
import datetime

# Directories Setup
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)


def run_ground_truth(config, out_dir):
    """Run MATLAB ground truth simulation, load data, and plot results."""
    setup_log("Running MATLAB ground truth simulation.")
    label = "ground_truth"
    dict_exp = config.experiment
    output_dir_gt, config_matlab = co.set_run(out_dir, config, label)
    uu.run_matlab_ground_truth()
    system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config_matlab, path=out_dir)


    # uu.compute_metrics([system_gt, *observers_gt, mm_obs_gt], config, out_dir)

    if config.experiment.plot:
        if config.model_parameters.n_obs == 0:
            system_meas, _ = uu.import_testdata(config)
            pp.plot_multiple_series([system_gt, system_meas], out_dir, label)

        elif config.model_parameters.n_obs == 1:
            pp.plot_multiple_series([system_gt, *observers_gt], out_dir, label)
            pp.plot_l2(system_gt, [*observers_gt], out_dir, label)
            pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], mm_obs_gt["theta"], out_dir, label)
            pp.plot_obs_err([*observers_gt], out_dir, label)

        else:
            if config.plot.show_obs:
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
    setup_log("Running simulation for the system.")
    label = "simulation_system"
    output_dir_system, cfg_system = co.set_run(out_dir, config, label)
    pinns_sys = uu.train_model(cfg_system)
    system = uu.get_pred(pinns_sys, system_gt["grid"], out_dir, "system")
    [], system = uu.calculate_l2(system_gt, [], system)
    [], system = uu.compute_obs_err(system_gt, [], system)


    uu.compute_metrics([system_gt, system], config, out_dir)

    if config.experiment.plot:
        pp.plot_multiple_series([system_gt, system], out_dir, label)
        pp.plot_l2(system_gt, [system], out_dir, label)
        pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], system["theta"], out_dir, label)
        pp.plot_obs_err([system], out_dir, label)


def run_simulation_inverse(config, out_dir, system_gt):
    # out_dir = "/home/guglielmo/pinns-bioheat-experiments/outputs/2025-02-10/11-31-24"
    """Run simulation for the system and plot results."""
    setup_log("Running simulation for the inverse problem.")
    label = "inverse"
    output_dir_system, cfg_inverse = co.set_run(out_dir, config, label)
    model, W = uu.create_model(cfg_inverse)
    props, pars = config.model_properties, config.model_parameters

    model.compile(
    "adam", lr=props.learning_rate, external_trainable_variables=W, loss_weights=[1, 1, 20]
    )
    variable1 = dde.callbacks.VariableValue(W, period=200, filename=f"{out_dir}/variable_W_adam.txt")

    losshistory, train_state = model.train(iterations=props.iters, callbacks=[variable1], model_save_path=f"{out_dir}/model.pt")

    model.compile(
    "L-BFGS", external_trainable_variables=W, loss_weights=[1, 1, 20]
    )
    variable1 = dde.callbacks.VariableValue(W, period=200, filename=f"{out_dir}/variable_W_lbfgs.txt")

    losshistory, train_state = model.train(iterations=props.iters_lbfgs, callbacks=[variable1], model_save_path=f"{out_dir}/model.pt")

    # dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=out_dir)

    # pinns_sys = uu.train_model(cfg_inverse)
    system = uu.get_pred(model, system_gt["grid"], out_dir, "system")
    [], system = uu.calculate_l2(system_gt, [], system)
    [], system = uu.compute_obs_err(system_gt, [], system)

    data1 = np.loadtxt(f"{out_dir}/variable_W_adam.txt", delimiter=' ', converters={1: lambda s: float(s.strip('[]'))})
    iters1 = np.array(data1[:, 0]).reshape(len(data1), 1)
    values1 = np.array(data1[:, 1]).reshape(len(data1), 1)
    data2 = np.loadtxt(f"{out_dir}/variable_W_lbfgs.txt", delimiter=' ', converters={1: lambda s: float(s.strip('[]'))})
    iters2 = np.array(data2[:, 0]).reshape(len(data2), 1)
    values2 = np.array(data2[:, 1]).reshape(len(data2), 1)
    iters = np.concatenate((iters1, iters2), axis=0)
    values = np.concatenate((values1, values2), axis=0)

    # true = np.full_like(values, config.model_parameters.W_sys)

    # uu.compute_metrics([system_gt, system], config, out_dir)

    if config.experiment.plot:
        pp.plot_multiple_series([system_gt, system], out_dir, label)
        pp.plot_l2(system_gt, [system], out_dir, label)
        pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], system["theta"], out_dir, label)
        pp.plot_obs_err([system], out_dir, label)
        pp.plot_loss_components(np.array(losshistory.loss_train), np.array(losshistory.loss_test), np.array(losshistory.steps), "inverse", fold=out_dir)
        # pp.plot_generic(x=[iters, iters], y=[values, true], title="Recovered W value", xlabel="Iterations", ylabel=r"$W \quad [s^{-1}]$", legend_labels=["PINNs", "MATLAB"], log_scale=True, log_xscale=False, 
        #     #  size=(6, 5), filename=f"{out_dir}/variable_W.png", colors=["cornflowerblue", "lightsteelblue"], linestyles=["-", ":"], markers=None,
        #     size=(6, 5), filename=f"{out_dir}/variable_W.png", colors=["cornflowerblue", "lightsteelblue"], linestyles=["-", ":"], markers=None,
        #      linewidths=None, markersizes=None, alphas=None, markevery=50)
        pp.plot_generic(x=[iters], y=[values], title="Recovered W value", xlabel="Iterations", ylabel=r"$W \quad [s^{-1}]$", legend_labels=["PINNs"], log_scale=True, log_xscale=False, 
            size=(6, 5), filename=f"{out_dir}/variable_W.png", colors=["cornflowerblue"], linestyles=["-"], markers=None,
             linewidths=None, markersizes=None, alphas=None, markevery=50)



def run_simulation_mm_obs(config, out_dir, system_gt, mm_obs_gt, observers_gt, gt_path=None):
    """Run multi-observer simulation, load data, and plot results."""
    setup_log("Running simulation for multi-observer.")
    label = "simulation_mm_obs"
    output_dir_inverse, config_inverse = co.set_run(out_dir, config, label)
    props, pars, exp = config_inverse.model_properties, config_inverse.model_parameters, config.experiment
    config_wb = {
        "num_domain": props.num_domain,
        "num_boundary": props.num_boundary,
        # "n_anchor_points": props.n_anchor_points,
        "alfa": props.alfa,
        "obs": pars.W_index,
    }
    if exp.wandb:
        wandb.init(project=f"{datetime.date.today()}_{exp.wandb_name}", config=config_wb)
    output = uu.execute(config_inverse, label)
    multi_obs = output[0] if pars.n_obs==1 else [e[0] for e in output]
    test_loss = output[1] if pars.n_obs==1 else [e[1] for e in output]
    x_obs = uu.gen_obsdata(config_inverse, system_gt)
    observers, mm_obs = uu.get_observers_preds(mm_obs_gt, multi_obs, x_obs, out_dir, config_inverse, label)

    metrics = uu.compute_metrics([mm_obs_gt, mm_obs], config, out_dir)
    metrics = {key.replace(f"observer_{pars.W_index}_", ""): value for key, value in metrics.items()}
    metrics["test"] = test_loss
    if exp.wandb:
        wandb.log(metrics)

    if config.experiment.plot:

        if config.plot.show_obs:
            if 1 < config.model_parameters.n_obs <= 8:
                    pp.plot_weights([*observers], out_dir, label)
            if config.plot.show_gt:
                if config.plot.show_sys:
                    pp.plot_multiple_series([system_gt, *observers, mm_obs_gt, mm_obs], out_dir, label)
                    pp.plot_l2(system_gt, [*observers, mm_obs_gt, mm_obs], out_dir, label)
                    pp.plot_obs_err([*observers, mm_obs_gt, mm_obs], out_dir, label)
                else:
                    pp.plot_multiple_series([*observers, mm_obs_gt, mm_obs], out_dir, label)
                    pp.plot_l2(system_gt, [*observers, mm_obs_gt, mm_obs], out_dir, label)
                    pp.plot_obs_err([*observers, mm_obs_gt, mm_obs], out_dir, label)
            else:
                if config.plot.show_sys:
                    pp.plot_multiple_series([system_gt, *observers, mm_obs], out_dir, label)
                    pp.plot_l2(system_gt, [*observers, mm_obs], out_dir, label)
                    pp.plot_obs_err([*observers, mm_obs], out_dir, label)
                else:
                    pp.plot_multiple_series([*observers, mm_obs], out_dir, label)
                    pp.plot_l2(system_gt, [*observers, mm_obs], out_dir, label)
                    pp.plot_obs_err([*observers, mm_obs], out_dir, label)
        else:
            if config.plot.show_gt:
                if config.plot.show_sys:
                    pp.plot_multiple_series([system_gt, mm_obs_gt, mm_obs], out_dir, label)
                    pp.plot_l2(system_gt, [mm_obs_gt, mm_obs], out_dir, label)
                    pp.plot_obs_err([mm_obs_gt, mm_obs], out_dir, label)
                else:
                    pp.plot_multiple_series([mm_obs_gt, mm_obs], out_dir, label)
                    pp.plot_l2(system_gt, [mm_obs_gt, mm_obs], out_dir, label)
                    pp.plot_obs_err([mm_obs_gt, mm_obs], out_dir, label)
            else:
                if config.plot.show_sys:
                    pp.plot_multiple_series([system_gt, mm_obs], out_dir, label)
                    pp.plot_l2(system_gt, [mm_obs], out_dir, label)
                    pp.plot_obs_err([mm_obs], out_dir, label)
                else:
                    pp.plot_multiple_series([mm_obs], out_dir, label)
                    pp.plot_l2(system_gt, [mm_obs], out_dir, label)
                    pp.plot_obs_err([mm_obs], out_dir, label)

        pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], mm_obs["theta"], out_dir, label)



def run_measurement_mm_obs(config, out_dir):
    """Run multi-observer simulation, load data, and plot results."""
    label = config.experiment.run
    setup_log(f"Running measurement {label} for multi-observer")
    output_dir_meas, config_meas = co.set_run(out_dir, config, label)
    multi_obs = uu.execute(config_meas, label)
    system_meas, _ = uu.import_testdata(config_meas)
    x_obs = uu.import_obsdata(config_meas)
    observers, mm_obs = uu.get_observers_preds(system_meas, multi_obs, x_obs, out_dir, config_meas, label)
    # load_dir = f"{tests_dir}/meas_cool_bone_tum_{config.model_parameters.n_obs}obs/0" if label.endswith("1") else f"{tests_dir}/meas_cool_bone_tum_{config.model_parameters.n_obs}obs/1"
    # observers, mm_obs = uu.load_observers_preds(load_dir, config_meas, label)

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

    gt_path=f"{tests_dir}/{dict_exp.gt_path}"

    if dict_exp["simulation"]:
        # Simulation System
        if n_ins==2 and not dict_exp["inverse"]:
            system_gt, _, _ = uu.gen_testdata(config, path=gt_path)
            run_simulation_system(config, run_out_dir, system_gt)
        
        elif n_ins==2 and dict_exp["inverse"]:
            if dict_exp["run"].startswith("meas"):
                system_gt, _ = uu.import_testdata(config)
            else:
                system_gt, _, _ = uu.gen_testdata(config, path=gt_path)
            run_simulation_inverse(config, run_out_dir, system_gt)

        # Simulation Multi-Observer
        else:
            if dict_exp["ground_truth"]:
                output_dir_gt, system_gt, observers_gt, mm_obs_gt = run_ground_truth(config, run_out_dir)
            else:
                system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config, path=gt_path)

            run_simulation_mm_obs(config, run_out_dir, system_gt, mm_obs_gt, observers_gt, gt_path)
    
    # elif dict_exp["run"].startswith("meas"):
    #     if dict_exp["ground_truth"]:
    #         output_dir_gt, system_gt, observers_gt, mm_obs_gt = run_ground_truth(config, run_out_dir)
            
    #     run_measurement_mm_obs(config, run_out_dir)


if __name__ == "__main__":
    main()