import os, logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import utils as uu
import common as co
import plots as pp
import deepxde as dde
import wandb
import datetime

# Directories Setup
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
gt_dir = os.path.join(git_dir, "gt")
os.makedirs(tests_dir, exist_ok=True)


def run_ground_truth(config, out_dir):
    """Run MATLAB ground truth simulation, load data, and plot results."""
    # setup_log("Running MATLAB ground truth simulation.")
    label = "simulation_ground_truth"
    output_dir_gt, config_matlab = co.set_run(out_dir, config, label)
    matlab_data = OmegaConf.create({
        "pdecoeff": config_matlab.pdecoeff,
        "parameters": config_matlab.parameters
        })
    
    matlab_hash = co.generate_config_hash(matlab_data)
    gt_path = f"{gt_dir}/gt_{matlab_hash}.txt"
    matlab_data.gt_path = gt_path
    OmegaConf.save(matlab_data, f"{conf_dir}/config_ground_truth.yaml")
    OmegaConf.save(matlab_data, f"{gt_dir}/cfg_{matlab_hash}.yaml")

    if not os.path.exists(gt_path):
        uu.run_matlab_ground_truth()

    system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config_matlab)
    pars = config.parameters

    # uu.compute_metrics([system_gt, *observers_gt, mm_obs_gt], config, out_dir)

    if config.plot.show:
        if pars.nobs == 0:
            system_meas, _ = uu.import_testdata(config)
            pp.plot_multiple_series([system_gt, system_meas], out_dir, label)

        elif pars.nobs == 1:
            pp.plot_multiple_series([system_gt, *observers_gt], out_dir, label)
            pp.plot_l2(system_gt, [*observers_gt], out_dir, label)
            pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], mm_obs_gt["theta"], out_dir, label)
            # pp.plot_obs_err(mm_obs_gt, out_dir, label)

        else:
            if config.plot.show_obs:
                pp.plot_multiple_series([system_gt, *observers_gt, mm_obs_gt], out_dir, label)
                pp.plot_l2(system_gt, [*observers_gt, mm_obs_gt], out_dir, label)
                pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], mm_obs_gt["theta"], out_dir, label)
                pp.plot_obs_err([*observers_gt, mm_obs_gt], out_dir, label)
                if 1 < pars.nobs <= 8:
                    pp.plot_weights([*observers_gt], out_dir, label)
        
            else:
                pp.plot_multiple_series([system_gt, mm_obs_gt], out_dir, label)
                pp.plot_l2(system_gt, [mm_obs_gt], out_dir, label)
                pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], mm_obs_gt["theta"], out_dir, label)
                pp.plot_obs_err([mm_obs_gt], out_dir, label)

        # if dict_exp["run"].startswith("meas"):
        #     system_meas, _ = uu.import_testdata(config)
        #     pp.plot_timeseries_with_predictions(system_meas, system_gt, config, out_dir)     
        
    
    return output_dir_gt, system_gt, observers_gt, mm_obs_gt


def run_simulation_system(config, out_dir, system_gt):
    """Run simulation for the system and plot results."""
    # setup_log("Running simulation for the system.")
    label = "simulation_system"
    props, exp = config.model_properties, config.experiment
    output_dir_system, cfg_system = co.set_run(out_dir, config, label)
    config_wb = {
    "num_domain": props.num_domain,
    "num_boundary": props.num_boundary,
    "resampling": props.resampling,
    }
    if exp.wandb:
        wandb.init(project=f"{datetime.date.today()}_{exp.wandb_name}", config=config_wb)
    pinns_sys, train_info = uu.train_model(cfg_system)
    system = uu.get_pred(pinns_sys, system_gt["grid"], out_dir, "system")
    [], system = uu.calculate_l2(system_gt, [], system)
    [], system = uu.compute_obs_err(system_gt, [], system)

    metrics = uu.compute_metrics([system_gt, system], train_info, config, out_dir)
    if exp.wandb:
        wandb.log(metrics)

    if config.experiment.plot:
        pp.plot_multiple_series([system_gt, system], out_dir, label)
        pp.plot_l2(system_gt, [system], out_dir, label)
        pp.plot_validation_3d(system_gt["grid"], system_gt["theta"], system["theta"], out_dir, label)
        pp.plot_obs_err([system], out_dir, label)


def run_simulation_inverse(config, out_dir, system_gt):
    # out_dir = "/home/guglielmo/pinns-bioheat-experiments/outputs/2025-02-10/11-31-24"
    """Run simulation for the system and plot results."""
    # setup_log("Running simulation for the inverse problem.")
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
    # setup_log("Running simulation for multi-observer.")
    label = "simulation_mm_obs"
    _, cfg_sim = co.set_run(out_dir, config, label)
    pdecoeff, hp, pars, exp = cfg_sim.pdecoeff,cfg_sim.hp, cfg_sim.parameters, config.experiment
    nobs = pars.nobs
    config_wb = {
        # "num_domain": props.num_domain,
        # "num_boundary": props.num_boundary,
        # "resampling": props.resampling,
        "oig": pdecoeff.oig
    }
    if exp.wandb:
        wandb.init(project=f"{datetime.date.today()}_{exp.wandb_name}", config=config_wb)
    output = uu.execute(cfg_sim, label)
    multi_obs = output[0] if nobs==1 else [e[0] for e in output]
    # train_info = output[1] if nobs==1 else [e[1] for e in output]
    x_obs = uu.gen_obsdata(cfg_sim, system_gt)
    observers, mm_obs = uu.get_observers_preds(mm_obs_gt, multi_obs, x_obs, out_dir, cfg_sim, label)

    metrics = uu.compute_metrics([mm_obs_gt, mm_obs], {}, config, out_dir)
    if exp.wandb:
        wandb.log(metrics)

    if config.plot.show:

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
    # setup_log(f"Running measurement {label} for multi-observer")
    output_dir_meas, config_meas = co.set_run(out_dir, config, label)
    # multi_obs = uu.execute(config_meas, label)
    system_meas, _ = uu.import_testdata(config_meas)
    # x_obs = uu.import_obsdata(config_meas)
    # observers, mm_obs = uu.get_observers_preds(system_meas, multi_obs, x_obs, out_dir, config_meas, label)
    # load_dir = f"{tests_dir}/meas_cool_bone_tum_{config.model_parameters.n_obs}obs/0" if label.endswith("1") else f"{tests_dir}/meas_cool_bone_tum_{config.model_parameters.n_obs}obs/1"
    # observers, mm_obs = uu.load_observers_preds(load_dir, config_meas, label)
    load_path = f"/Users/guglielmocappellini/Desktop/research/code/pinns-bioheat-experiments/tests/meas_cool_bone_tum_64obs/1/multi_observer_{label}.txt"
    data = np.loadtxt(load_path)
    x, t, sys = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T
    X = np.vstack((x, t)).T
    y_sys = sys.flatten()[:, None]
    mm_obs = {"grid": X, "theta": y_sys, "label": "multi_observer"}
    observers = []

    # uu.compute_metrics([system_meas, mm_obs], {}, config, out_dir)

    if config.plot.show:

        if config.plot.show_obs:
            pp.plot_multiple_series([system_meas, *observers, mm_obs], out_dir, label)
            # pp.plot_l2(system_meas, [*observers, mm_obs], out_dir, label)
            # pp.plot_validation_3d(system_meas["grid"], system_meas["theta"], mm_obs["theta"], out_dir, label)
            # pp.plot_obs_err([*observers, mm_obs], out_dir, label)
            # if 1 < config.model_parameters.n_obs <= 8: 
            #     pp.plot_weights([*observers], out_dir, label)
        else:
            pp.plot_multiple_series([system_meas, mm_obs], out_dir, label)
            # pp.plot_l2(system_meas, [mm_obs], out_dir, label)
            # pp.plot_validation_3d(system_meas["grid"], system_meas["theta"], mm_obs["theta"], out_dir, label)
            # pp.plot_obs_err([mm_obs], out_dir, label)

        pp.plot_timeseries_with_predictions(system_meas, mm_obs, config, out_dir)

# @hydra.main(version_base=None, config_path=conf_dir, config_name="config_run")
def main(config: DictConfig):
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """

    # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # config.output_dir = output_dir
    # print(f'Working dir: {output_dir}')

    run_out_dir = config.output_dir
    dict_exp = config.experiment
    nins = config.hp.nins

    # gt_path=f"{tests_dir}/{dict_exp.gt_path}"

    if dict_exp["simulation"]:
        # Simulation System
        if nins==2 and not dict_exp["inverse"]:
            system_gt, _, _ = uu.gen_testdata(config)
            run_simulation_system(config, run_out_dir, system_gt)
        
        elif nins==2 and dict_exp["inverse"]:
            if dict_exp["run"].startswith("meas"):
                system_gt, _ = uu.import_testdata(config)
            else:
                system_gt, _, _ = uu.gen_testdata(config)
            run_simulation_inverse(config, run_out_dir, system_gt)

        # Simulation Multi-Observer
        else:
            if dict_exp["ground_truth"]:
                output_dir_gt, system_gt, observers_gt, mm_obs_gt = run_ground_truth(config, run_out_dir)
            else:
                system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config)

            run_simulation_mm_obs(config, run_out_dir, system_gt, mm_obs_gt, observers_gt)
    
    elif dict_exp["run"].startswith("meas"):
        if dict_exp["ground_truth"]:
            output_dir_gt, system_gt, observers_gt, mm_obs_gt = run_ground_truth(config, run_out_dir)
            
        run_measurement_mm_obs(config, run_out_dir)


if __name__ == "__main__":
    initialize('./configs', version_base=None) 
    conf = compose(config_name='config_run')
    main(conf)