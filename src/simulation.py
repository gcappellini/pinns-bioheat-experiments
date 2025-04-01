import os
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
    
    # matlab_hash = co.generate_config_hash(matlab_data)
    # gt_path = f"{gt_dir}/gt_{matlab_hash}"
    # matlab_data.gt_path = gt_path
    # OmegaConf.save(matlab_data, f"{conf_dir}/config_ground_truth.yaml")
    # OmegaConf.save(matlab_data, f"{gt_dir}/cfg_{matlab_hash}.yaml")

    # if not os.path.exists(f"{gt_path}.txt"):
    #     uu.run_matlab_ground_truth()

    config_matlab.experiment.gt_path = f"{tests_dir}/cooling_ground_truth_5e-04/output_matlab_8Obs"
    system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config_matlab)

    if matlab_data.parameters.nobs==0:
        return system_gt, observers_gt, mm_obs_gt
    else:
        observers_gt, mm_obs_gt = uu.compute_obs_err(system_gt, observers_gt, mm_obs_gt) 
        return system_gt, observers_gt, mm_obs_gt



def run_simulation_system(config, out_dir, system_gt):
    """Run simulation for the system and plot results."""
    # setup_log("Running simulation for the system.")
    label = "simulation_system"
    props, exp = config.hp, config.experiment
    output_dir_system, cfg_system = co.set_run(out_dir, config, label)

    cfg_wandb = OmegaConf.to_container(config.pdecoeff)
    if exp.wandb:
        wandb.init(project=f"{datetime.date.today()}_{exp.wandb_name}", config=cfg_wandb)
    pinns_sys, train_info = uu.train_model(cfg_system)
    system = uu.get_pred(pinns_sys, system_gt["grid"], out_dir, "system")
    [], system = uu.calculate_l2(system_gt, [], system)
    [], system = uu.compute_obs_err(system_gt, [], system)

    metrics = uu.compute_metrics([system_gt, system], train_info, config, out_dir)
    if exp.wandb:
        wandb.log(metrics)
    
    return system


def run_simulation_inverse(config, out_dir, system_gt):
    """Run simulation for the system and plot results."""
    # setup_log("Running simulation for the inverse problem.")
    label = "inverse"
    output_dir_system, cfg_inverse = co.set_run(out_dir, config, label)
    model, wbinv = uu.create_model(cfg_inverse)
    hp, pars = config.hp, config.parameters

    model.compile(
        "adam", lr=hp.lr, external_trainable_variables=wbinv, loss_weights=[1, 1, 20]
    )
    variable1 = dde.callbacks.VariableValue(wbinv, period=200, filename=f"{out_dir}/variable_wbinv_adam.txt")
    losshistory, train_state = model.train(iterations=hp.iters, callbacks=[variable1], model_save_path=f"{out_dir}/model.pt")

    model.compile(
    "L-BFGS", external_trainable_variables=wbinv, loss_weights=[1, 1, 20]
    )
    variable1 = dde.callbacks.VariableValue(wbinv, period=200, filename=f"{out_dir}/variable_wbinv_lbfgs.txt")
    losshistory, train_state = model.train(iterations=hp.iters_lbfgs, callbacks=[variable1], model_save_path=f"{out_dir}/model.pt")
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=out_dir)

    # pinns_sys = uu.train_model(cfg_inverse)
    system = uu.get_pred(model, system_gt["grid"], out_dir, "system")
    [], system = uu.calculate_l2(system_gt, [], system)
    [], system = uu.compute_obs_err(system_gt, [], system)

    data1 = np.loadtxt(f"{out_dir}/variable_wbinv_adam.txt", delimiter=' ', converters={1: lambda s: float(s.strip('[]'))})
    iters1 = np.array(data1[:, 0]).reshape(len(data1), 1)
    values1 = np.array(data1[:, 1]).reshape(len(data1), 1)
    data2 = np.loadtxt(f"{out_dir}/variable_wbinv_lbfgs.txt", delimiter=' ', converters={1: lambda s: float(s.strip('[]'))})
    iters2 = np.array(data2[:, 0]).reshape(len(data2), 1)
    values2 = np.array(data2[:, 1]).reshape(len(data2), 1)
    iters = np.concatenate((iters1, iters2), axis=0)
    values = np.concatenate((values1, values2), axis=0)

    uu.compute_metrics([system_gt, system], config, out_dir)

    uu.compute_metrics([system_gt, system], config, out_dir)
    inv_perf = {"iters": iters, "values": values, "label": "wbinv"}
    pp.plot_loss_components(np.array(losshistory.loss_train), np.array(losshistory.loss_test), np.array(losshistory.steps), "inverse", fold=out_dir)
    return system, inv_perf


def run_simulation_mm_obs(config, out_dir, system_gt, mm_obs_gt, observers_gt, gt_path=None):
    """Run multi-observer simulation, load data, and plot results."""
    # setup_log("Running simulation for multi-observer.")
    label = "simulation_mm_obs"
    _, cfg_sim = co.set_run(out_dir, config, label)
    cfg_sim.experiment.pred_fold = f"{tests_dir}/cooling_simulation_5e-04"
    pdecoeff, hp, pars, exp = cfg_sim.pdecoeff,cfg_sim.hp, cfg_sim.parameters, config.experiment
    nobs = pars.nobs
    if exp.wandb:
        wandb.init(project=f"{datetime.date.today()}_{exp.wandb_name}", config=config.hp)
    if conf.experiment.pred_fold is None:
        output = uu.execute(cfg_sim, label)
        multi_obs = output[0] if nobs==1 else [e[0] for e in output]
        # train_info = output[1] if nobs==1 else [e[1] for e in output]
        x_obs = uu.gen_obsdata(cfg_sim, system_gt)
        observers, mm_obs = uu.get_observers_preds(mm_obs_gt, multi_obs, x_obs, out_dir, cfg_sim, label)
    else:
        observers, mm_obs = uu.load_observers_preds(system_gt, cfg_sim, label)
    metrics = uu.compute_metrics([mm_obs_gt, mm_obs], None, config, out_dir)

    if exp.wandb:
        wandb.log(metrics)

    return observers, mm_obs


def run_measurement(config, out_dir):
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

    # load_path = f"/Users/guglielmocappellini/Desktop/research/code/pinns-bioheat-experiments/tests/meas_cool_bone_tum_16obs/0/multi_observer_{label}.txt"
    # data = np.loadtxt(load_path)
    # x, t, sys = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T
    # X = np.vstack((x, t)).T
    # y_sys = sys.flatten()[:, None]
    # mm_obs = {"grid": X, "theta": y_sys, "label": "multi_observer"}
    # observers = {}
    # obs_dict={}
    # _, mm_obs = uu.compute_obs_err(system_meas, obs_dict, mm_obs)
    # _, mm_obs = uu.calculate_l2(system_meas, obs_dict, mm_obs)
    nobs = config.parameters.nobs
    exp_name = 0 if label.endswith("1") else 1
    if nobs == 8:
        config_meas.plot.show_obs = True
    else:
        config_meas.plot.show_obs = False


    config_meas.experiment.pred_fold = f"{tests_dir}//meas_cool_bone_tum_{nobs}obs/{exp_name}"
    observers, mm_obs = uu.load_observers_preds(system_meas, config_meas, label)


    # uu.compute_metrics([system_meas, mm_obs], {}, config, out_dir)
    if config.plot.show:
        pp.plot_res(config, system_meas=system_meas, observers=observers, mm_obs=mm_obs)
    return observers, mm_obs


def run_simulation(config, out_dir):
    """Determines the type of simulation and runs it accordingly."""
    system_gt, observers_gt, mm_obs_gt = run_ground_truth(config, out_dir)
    system, observers, mm_obs, var = None, None, None, None

    if config.hp.nins == 2:
        if config.experiment.inverse:
            if config.experiment.run.startswith("meas"):
                system_gt, _ = uu.import_testdata(config)
            system, var = run_simulation_inverse(config, out_dir, system_gt)
        else:
            system = run_simulation_system(config, out_dir, system_gt)
    else:
        observers, mm_obs = run_simulation_mm_obs(config, out_dir, system_gt, mm_obs_gt, observers_gt)
    
    if config.plot.show:
        pp.plot_res(config, system_gt=system_gt, system=system, observers=observers, observers_gt=observers_gt, mm_obs_gt=mm_obs_gt, mm_obs=mm_obs, var=var)


def main(config: DictConfig):
    """Main function to run simulations, measurements, and ground truth processing."""
    if config.experiment.run.startswith("ground_truth"):
        system_gt, observers_gt, mm_obs_gt = run_ground_truth(config, config.output_dir)
        if config.plot.show:
            pp.plot_res(config, system_gt=system_gt, observers_gt=observers_gt, mm_obs_gt=mm_obs_gt )
    elif config.experiment.run.startswith("simulation"):
        run_simulation(config, config.output_dir)
    elif config.experiment.run.startswith("meas"):
        run_measurement(config, config.output_dir)


if __name__ == "__main__":
    initialize('./configs', version_base=None) 
    conf = compose(config_name='config_run')
    main(conf)