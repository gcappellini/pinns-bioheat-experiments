import deepxde as dde
import numpy as np
import os
import torch
import wandb
from scipy.interpolate import interp1d
from scipy import integrate
import pickle
import datetime
import pandas as pd
import coeff_calc as cc
import plots as pp
import common as co
from omegaconf import OmegaConf
import matlab.engine
import subprocess


dde.config.set_random_seed(200)

dev = torch.device("cpu")
# dev = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
conf_dir = os.path.join(src_dir, "configs")


models = os.path.join(git_dir, "models")
os.makedirs(models, exist_ok=True)

f1, f2, f3 = [None]*3
n_digits = 6


def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(0)
    return losshistory.loss_train[0]


def load_from_pickle(file_path):
    with open(file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)
    

def compute_metrics(grid, true, pred, run_figs, system=None):
    # Load loss weights from configuration
    cfg = OmegaConf.load(f"{run_figs}/config.yaml")
    loss_weights = cfg.model_parameters.loss_weights
    small_number = 1e-8
    
    true = np.ravel(true)
    pred = np.ravel(pred)
    true_nonzero = np.where(true != 0, true, small_number)
    
    # Part 1: General metrics for pred (Observer PINNs vs Observer MATLAB)
    L2RE = np.sum(calculate_l2(grid, true, pred))
    MSE = calculate_mse(true, pred)
    max_err = np.max(np.abs(true_nonzero - pred))
    mean_err = np.mean(np.abs(true_nonzero - pred))
    
    metrics = {
        "total_L2RE": L2RE,
        "total_MSE": MSE,
        "total_max": max_err,
        "total_mean": mean_err,
    }
    
    # If system predictions are provided, calculate metrics for system (Observer PINNs vs System MATLAB)
    if system is not None:
        system = np.ravel(system)
        system_nonzero = np.where(system != 0, system, small_number)
        
        L2RE_sys = np.sum(calculate_l2(grid, pred, system))
        MSE_sys = calculate_mse(true, system)
        max_err_sys = np.max(np.abs(system_nonzero - pred))
        mean_err_sys = np.mean(np.abs(system_nonzero - pred))
        
        # Store total metrics for system
        metrics.update({
            "total_L2RE_sys": L2RE_sys,
            "total_MSE_sys": MSE_sys,
            "total_max_sys": max_err_sys,
            "total_mean_sys": mean_err_sys,
        })

    # Define conditions for initial, left boundary, right boundary, and domain
    conditions = {
        "initial_condition": grid[:, 1] == 0,  # Time t = 0
        "bc0": grid[:, 0] == 0,                # Left boundary x = 0
        "bc1": grid[:, 0] == 1,                # Right boundary x = 1
    }
    domain_condition = ~(conditions["initial_condition"] | conditions["bc0"] | conditions["bc1"])
    conditions["domain"] = domain_condition

    # Compute metrics for each condition and add to metrics dictionary
    for cond_name, condition in conditions.items():
        # Metrics for pred
        true_cond = true[condition]
        pred_cond = pred[condition]
        true_nonzero_cond = np.where(true_cond != 0, true_cond, small_number)
        
        # Calculate metrics for pred under this specific condition
        L2RE_cond = np.sum(calculate_l2(grid[condition], true_cond, pred_cond))
        MSE_cond = calculate_mse(true_cond, pred_cond)
        max_err_cond = np.max(np.abs(true_nonzero_cond - pred_cond))
        mean_err_cond = np.mean(np.abs(true_nonzero_cond - pred_cond))
        
        # Store these metrics in the dictionary
        metrics.update({
            f"{cond_name}_L2RE": L2RE_cond,
            f"{cond_name}_MSE": MSE_cond,
            f"{cond_name}_max": max_err_cond,
            f"{cond_name}_mean": mean_err_cond,
        })
        
        # Metrics for system, if provided
        if system is not None:
            system_cond = system[condition]
            system_nonzero_cond = np.where(system_cond != 0, system_cond, small_number)
            
            # Calculate metrics for system under this specific condition
            L2RE_sys_cond = np.sum(calculate_l2(grid[condition], pred_cond, system_cond))
            MSE_sys_cond = calculate_mse(pred_cond, system_cond)
            max_err_sys_cond = np.max(np.abs(system_nonzero_cond - pred_cond))
            mean_err_sys_cond = np.mean(np.abs(system_nonzero_cond - pred_cond))
            
            # Store these system metrics in the dictionary
            metrics.update({
                f"{cond_name}_L2RE_sys": L2RE_sys_cond,
                f"{cond_name}_MSE_sys": MSE_sys_cond,
                f"{cond_name}_max_sys": max_err_sys_cond,
                f"{cond_name}_mean_sys": mean_err_sys_cond,
            })

    # Calculate and store LOSS metric for each condition (Observer PINNs vs Observer MATLAB)
    LOSS_pred = np.sum(loss_weights * np.array([metrics[f"{cond}_MSE"] for cond in ["domain", "bc0", "bc1", "initial_condition"]]))
    metrics["total_LOSS"] = LOSS_pred
    for cond_name in conditions.keys():
        metrics[f"{cond_name}_LOSS"] = loss_weights[0] * metrics.get(f"{cond_name}_MSE", 0)

    # Calculate and store LOSS metric for system if provided (Observer PINNs vs System MATLAB)
    if system is not None:
        LOSS_sys = np.sum(loss_weights * np.array([metrics[f"{cond}_MSE_sys"] for cond in ["domain", "bc0", "bc1", "initial_condition"]]))
        metrics["total_LOSS_sys"] = LOSS_sys
        for cond_name in conditions.keys():
            metrics[f"{cond_name}_LOSS_sys"] = loss_weights[0] * metrics.get(f"{cond_name}_MSE_sys", 0)

    # Write all metrics to file with improved formatting
    with open(f"{run_figs}/metrics.txt", "w") as file:
        file.write("=== METRICS REPORT ===\n\n")
        
        # Part 1: Observer PINNs vs Observer MATLAB
        file.write("Part 1: Observer PINNs vs Observer MATLAB\n\n")
        for metric_name in ["L2RE", "MSE", "max", "mean", "LOSS"]:
            file.write(f"{metric_name.upper()}:\n")
            file.write(f"  Total: {metrics[f'total_{metric_name}']}\n")
            for cond_name in conditions.keys():
                file.write(f"  {cond_name.capitalize()}: {metrics[f'{cond_name}_{metric_name}']}\n")
            file.write("\n")  # Blank line between sections
            
        # Part 2: Observer PINNs vs System MATLAB, if system is provided
        if system is not None:
            file.write("Part 2: Observer PINNs vs System MATLAB\n\n")
            for metric_name in ["L2RE_sys", "MSE_sys", "max_sys", "mean_sys", "LOSS_sys"]:
                file.write(f"{metric_name.split('_')[0].upper()}:\n")
                file.write(f"  Total: {metrics[f'total_{metric_name}']}\n")
                for cond_name in conditions.keys():
                    file.write(f"  {cond_name.capitalize()}: {metrics[f'{cond_name}_{metric_name}']}\n")
                file.write("\n")  # Blank line between sections for readability

    return metrics


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def output_transform(x, y):
    return x[:, 0:1] * y


def create_model(run_figs):
    """
    Generalized function to create and configure a PDE solver model using DeepXDE.
    
    :param run_figs: Path to the directory containing the configuration file.
    :param direct: Boolean flag to configure the model as "sys" (direct=True) or "nbho" (direct=False).
    :param src_dir: Optional source directory for saving the configuration.
    :return: A compiled DeepXDE model.
    """
    # Load configuration
    config = OmegaConf.load(f'{run_figs}/config.yaml')
    model_props = config.model_properties
    model_pars = config.model_parameters

    # Extract shared parameters from the configuration
    direct = model_props.direct
    activation = model_props.activation
    initial_weights_regularizer = model_props.initial_weights_regularizer
    initialization = model_props.initialization
    learning_rate = model_props.learning_rate
    num_dense_layers = model_props.num_dense_layers
    num_dense_nodes = model_props.num_dense_nodes
    loss_weights = [model_props.w_res, model_props.w_bc0, model_props.w_bc1, model_props.w_ic]
    num_domain, num_boundary, num_initial, num_test = (
        model_props.num_domain, model_props.num_boundary,
        model_props.num_initial, model_props.num_test,
    )
    a1, a2, a3, a4, a5 = cc.a1, cc.a2, cc.a3, cc.a4, cc.a5
    K = cc.K

    # Shared problem constants
    W = model_props.W if not direct else model_pars.W_sys
    theta10 = scale_t(model_props.Ty10)

    def ic_fun(x):
        z = x if len(x.shape) == 1 else x[:, :1]
        theta20 = scale_t(model_props.Ty20)
        theta30 = scale_t(model_props.Ty30)

        c_2 = a5 * (theta30 - theta20)
        c_3 = theta20

        if direct:
            c_1 = -c_2 - c_3
            return c_1 * z**2 + c_2 * z + c_3
        
        else:
            c_1 = model_props.b2 - (c_2 - K * c_3) / K
            return ((c_2 - K * c_3) / K + c_1 * np.exp(K * z)) * (1 - z) ** model_props.b1

    def bc0_fun(x, theta, _):
        y3 = scale_t(model_props.Ty30)
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
        if direct:
            return dtheta_x + a5 * (y3 - theta)
        else:
            y2 = x[:, 1:2]
            return dtheta_x - K * theta - (a5 - K) * y2 + a5 * 0

    def bc1_fun(x, theta, _):
        return theta - theta10


    # Shared PDE
    def pde(x, theta):
        dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1 if direct else 2)
        dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
        source_term = -a3 * torch.exp(-a4 * x[:, :1])
        return a1 * dtheta_tau - dtheta_xx + W * a2 * theta + source_term

    # Geometry and time domain
    if direct:
        geom = dde.geometry.Interval(0, 1)
    else:
        geom = dde.geometry.Rectangle([0, 0], [1, 1])

    timedomain = dde.geometry.TimeDomain(0, 1.5)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ic = dde.icbc.IC(geomtime, ic_fun, lambda _, on_initial: on_initial)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_fun, boundary_1)
    bc_0 = dde.icbc.OperatorBC(geomtime, bc0_fun, boundary_0)

    # Data object
    data = dde.data.TimePDE(
        geomtime,
        lambda x, theta: pde(x, theta),
        [bc_0, bc_1, ic],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
    )

    # Define the network
    input_dim = 2 if direct else 3
    layer_size = [input_dim] + [num_dense_nodes] * num_dense_layers + [1]
    net = dde.nn.FNN(layer_size, activation, initialization)

    # Compile the model
    model = dde.Model(data, net)
    if initial_weights_regularizer:
        initial_losses = get_initial_loss(model)
        loss_weights = [
            lw * len(initial_losses) / il
            for lw, il in zip(loss_weights, initial_losses)
        ]
        model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    else:
        model.compile("adam", lr=learning_rate, loss_weights=loss_weights)

    OmegaConf.save(config, f"{src_dir}/config.yaml")
    OmegaConf.save(config, f"{run_figs}/config.yaml")

    return model

def train_model(run_figs, system=False):
    global models
    config = OmegaConf.load(f'{run_figs}/config.yaml')
    config_hash = co.generate_config_hash(config.model_properties)
    n = config.model_properties.iterations
    model_path = os.path.join(models, f"model_{config_hash}.pt-{n}.pt")

    mm = create_model(run_figs)

    if os.path.exists(model_path):
        # Model exists, load it
        print(f"Loading model from {model_path}")
        mm.restore(model_path, device=torch.device(dev), verbose=0)
        return mm
    
    LBFGS = config.model_properties.LBFGS
    resampler = config.model_properties.resampling
    resampler_period = config.model_properties.resampler_period

    callbacks = [dde.callbacks.PDEPointResampler(period=resampler_period)] if resampler else []

    losshistory, mm = train_and_save_model(mm, callbacks, run_figs)

    if LBFGS:
        # if ini_w:
            # ini_w = config.model_properties.initial_weights_regularizer
        #     initial_losses = get_initial_loss(mm)
        #     loss_weights = len(initial_losses) / initial_losses
        #     mm.compile("L-BFGS", loss_weights=loss_weights)
        # else:
        mm.compile("L-BFGS")
        losshistory, mm = train_and_save_model(mm, callbacks, run_figs)
        
    pp.plot_loss_components(losshistory, config_hash)
    return mm


def train_and_save_model(model, callbacks, run_figs):
    global models

    conf = OmegaConf.load(f'{run_figs}/config.yaml')
    config_hash = co.generate_config_hash(conf.model_properties)
    model_path = os.path.join(models, f"model_{config_hash}.pt")

    losshistory, _ = model.train(
        iterations=conf.model_properties.iterations,
        callbacks=callbacks,
        model_save_path=model_path,
        display_every=conf.plot.display_every
    )
    confi_path = os.path.join(models, f"config_{config_hash}.yaml")
    OmegaConf.save(conf, confi_path)


    return losshistory, model


def gen_testdata(conf, path=None):
    n = conf.model_parameters.n_obs
    dir_name = path if path is not None else conf.output_dir
    # if hpo:
    #     output_folder = f"{tests_dir}/cooling_simulation/ground_truth"
    # else:
    #     output_folder = f"{tests_dir}/{dir_name}"

    file_path = f"{dir_name}/ground_truth/output_matlab_{n}Obs.txt"

    try:
        data = np.loadtxt(file_path)

        if n == 8:
            x, t, sys, y_obs, mmobs = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T, data[:, 3:11], data[:, 11:12].T
            y_mm_obs = mmobs.flatten()[:, None]
        elif n == 3:
            x, t, sys, y_obs, mmobs = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T, data[:, 3:6], data[:, 6:7].T
            y_mm_obs = mmobs.flatten()[:, None]
        elif n == 1:
            x, t, sys, y_obs = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T, data[:, 3:4].T
            y_obs = y_obs.flatten()[:, None]
            y_mm_obs = y_obs

    except FileNotFoundError:
        print(f"File not found: {file_path}.")

        # subprocess.run(["python", f"{src_dir}/main.py"], check=True)

    X = np.vstack((x, t)).T
    y_sys = sys.flatten()[:, None]
    
    return np.hstack((X, y_sys, y_obs, y_mm_obs))


def load_weights(conf):
    n = conf.model_parameters.n_obs
    name = conf.experiment.name
    output_folder = f"{tests_dir}/{name[0]}_{name[1]}/ground_truth"
    if n==8:
        data = np.loadtxt(f"{output_folder}/weights_matlab_{n}Obs.txt")
        t, weights = data[:, 0:1], data[:, 1:9].T
    if n==3:
        data = np.loadtxt(f"{output_folder}/weights_matlab_{n}Obs.txt")
        t, weights = data[:, 0:1], data[:, 1:4].T
    return t, np.array(weights)


def gen_obsdata(conf, path=None):
    global f1, f2, f3

    solution = gen_testdata(conf, path)
    g = solution[:, 0:3]

    # g = np.hstack((X, y_sys))
    instants = np.unique(g[:, 1])
    
    rows_1 = g[g[:, 0] == 1.0]
    rows_0 = g[g[:, 0] == 0.0]

    # y1 = rows_1[:, 2].reshape(len(instants),)
    # f1 = interp1d(instants, y1, kind='previous')

    y2 = rows_0[:, 2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    # y30 = scale_t(conf.model_properties.Ty30)
    # y3 = np.full_like(y2, y30)
    # f3 = interp1d(instants, y3, kind='previous')

    # Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    # Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), g[:, 1])).T
    Xobs = np.vstack((g[:, 0], f2(g[:, 1]), g[:, 1])).T
    return Xobs


def load_from_pickle(file_path):
    with open(file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)
    
def scale_t(t):
    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    Troom = properties.model_properties.Troom
    Tmax = properties.model_properties.Tmax
    k = (t - Troom) / (Tmax - Troom)

    return round(k, n_digits)

def rescale_t(theta):
    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    Troom = properties.model_properties.Troom
    Tmax = properties.model_properties.Tmax

    # Iterate through each component in theta and rescale if it is a list-like object
    rescaled_theta = []
    if isinstance(theta, (int, float)):
        part = np.array(theta, dtype=float)  # Ensure each part is converted into a numpy array
        rescaled_part = Troom + (Tmax - Troom) * part  # Apply the rescaling
        rescaled_theta.append(np.round(rescaled_part, n_digits)) 

    else:
        for part in theta:
            part = np.array(part, dtype=float)  # Ensure each part is converted into a numpy array
            rescaled_part = Troom + (Tmax - Troom) * part  # Apply the rescaling
            rescaled_theta.append(np.round(rescaled_part, n_digits))  # Round and append each rescaled part
    return rescaled_theta

def rescale_x(X):
    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    L0 = properties.model_properties.L0

    # Iterate through each component in X and rescale if it is a list-like object
    rescaled_X = []
    for part in X:
        part = np.array(part, dtype=float)  # Ensure each part is converted into a numpy array
        rescaled_part = part * L0           # Apply the scaling
        rescaled_X.append(rescaled_part)    # Append rescaled part to the result list

    return rescaled_X

def rescale_time(tau):
    tau = np.array(tau)
    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    tauf = properties.model_properties.tauf
    j = tau*tauf

    return np.round(j, 0)

def scale_time(t):
    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    tauf = properties.model_properties.tauf
    j = t/tauf

    return np.round(j, n_digits)

def get_tc_positions():
    daa = OmegaConf.load(f"{src_dir}/config.yaml")
    L0 = daa.model_properties.L0
    x_y2 = 0.0
    x_gt2 = (daa.model_parameters.x_gt2)/L0
    x_gt1 = (daa.model_parameters.x_gt1)/L0
    x_y1 = 1.0

    return [x_y2, round(x_gt2, 2), round(x_gt1, 2), x_y1] 

def import_testdata(name):
    df = load_from_pickle(f"{src_dir}/data/vessel/{name}.pkl")

    positions = get_tc_positions()
    dfs = []
    boluses = []
    for time_value in df['tau']:

        # Extract 'theta' values for the current 'time' from df_result
        theta_values = df[df['tau'] == time_value][
            ['y2', 'gt2', 'gt1', 'y1']].values.flatten()

        time_array = np.array([positions, [time_value] * 4, theta_values]).T
        bol_value = df[df['tau'] == time_value][['y3']].values.flatten()

        bolus_array = np.array([bol_value]*4)
        
        boluses.append(bolus_array)
        dfs.append(time_array)

    vstack_array = np.vstack(dfs)
    boluses_arr = np.vstack(boluses)

    return np.hstack((vstack_array,boluses_arr))


def import_obsdata(nam, extended = True):
    global f1, f2, f3
    g = import_testdata(nam)
    instants = np.unique(g[:, 1])

    positions = get_tc_positions()

    rows_1 = g[g[:, 0] == positions[-1]]
    rows_0 = g[g[:, 0] == positions[0]]

    # y1 = rows_1[:, -2].reshape(len(instants),)
    # f1 = interp1d(instants, y1, kind='previous')

    y2 = rows_0[:, -2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    # y3 = rows_0[:, -1].reshape(len(instants),)
    # y3 = np.zeros_like(y2)
    # f3 = interp1d(instants, y3, kind='previous')

    if extended:
        x = np.linspace(0, 1, 101)
        t = np.linspace(0, 1, 101)

        X, T = np.meshgrid(x, t)
        T_clipped = np.clip(T, None, 0.9956)
        # Xobs = np.vstack((np.ravel(X), f1(np.ravel(T_clipped)), f2(np.ravel(T_clipped)), f3(np.ravel(T_clipped)), np.ravel(T))).T
        # Xobs = np.vstack((np.ravel(X), f1(np.ravel(T_clipped)), f2(np.ravel(T_clipped)), np.ravel(T))).T
        Xobs = np.vstack((np.ravel(X), f2(np.ravel(T_clipped)), np.ravel(T))).T
    else:
        # Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
        # Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), g[:, 1])).T
        Xobs = np.vstack((g[:, 0], f2(g[:, 1]), g[:, 1])).T

    return Xobs


def mm_observer(config):

    n_obs = config.model_parameters.n_obs
    out_dir = config.output_dir
    simul_dir = os.path.join(out_dir, f"simulation_{n_obs}obs")

    if n_obs==8:
        W0, W1, W2, W3, W4, W5, W6, W7 = config.model_parameters.W0, config.model_parameters.W1, config.model_parameters.W2, config.model_parameters.W3, config.model_parameters.W4, config.model_parameters.W5, config.model_parameters.W6, config.model_parameters.W7
        obs = np.array([W0, W1, W2, W3, W4, W5, W6, W7])
    if n_obs==3:
        W0, W1, W2 = config.model_parameters.W0, config.model_parameters.W4, config.model_parameters.W7
        obs = np.array([W0, W1, W2])

    if n_obs==1:
        W = config.model_parameters.W_obs
        config.model_properties.W = float(W)
        run_figs = co.set_run(simul_dir, config, "obs_0")
  
        return train_model(run_figs)

    multi_obs = []
    
    for j in range(n_obs):
        perf = obs[j]
        config.model_properties.W = float(perf)
        run_figs = co.set_run(f"obs_{j}")
        OmegaConf.save(config, f"{run_figs}/config.yaml") 

        model = train_model(run_figs)
        multi_obs.append(model)


    return multi_obs


def mu(o, tau_in):
    global f1, f2, f3

    tau = np.where(tau_in<0.9944, tau_in, 0.9944)
    xo = np.vstack((np.zeros_like(tau), f1(tau), f2(tau), f3(tau), tau)).T
    muu = []

    for el in o:
        oss = el.predict(xo)
        true = f2(tau)
        scrt = calculate_mu(oss, true)
        muu.append(scrt)
    muu = np.column_stack(muu)#.reshape(len(muu),)
    return muu


def calculate_mu(os, tr):
    tr = tr.reshape(os.shape)
    net = OmegaConf.load(f"{src_dir}/config.yaml")
    upsilon = net.model_parameters.upsilon
    scrt = upsilon*np.abs(os-tr)**2
    return scrt


def compute_mu(conf):
    n_obs = conf.model_parameters.n_obs
    g = np.hstack((gen_testdata(conf)))
    rows_0 = g[g[:, 0] == 0.0]
    sys_0 = rows_0[:, 2:3]
    if n_obs==8:
        obss_0 = rows_0[:, 3:11]
    if n_obs==3:
        obss_0 = rows_0[:, 3:6]

    muu = []

    for el in range(obss_0.shape[1]):
        mu_value = calculate_mu(obss_0[:, el], sys_0)
        muu.append(mu_value)
    muu = np.column_stack(muu)#.reshape(len(muu),)
    return muu



def mm_predict(multi_obs, obs_grid, prj_figs):
    conf = OmegaConf.load(f"{prj_figs}/config.yaml")
    ups = conf.model_parameters.upsilon
    lam = conf.model_parameters.lam
    a = np.load(f'{prj_figs}/weights_l_{lam}_u_{ups}.npy', allow_pickle=True)
    weights = a[1:]

    num_time_steps = weights.shape[1]

    predictions = []

    for row in obs_grid:
        t = row[-1]
        closest_idx = int(np.round(t * (num_time_steps - 1)))
        w = weights[:, closest_idx]

        # Predict using the multi_obs predictors for the current row
        o_preds = np.array([multi_obs[i].predict(row.reshape(1, -1)) for i in range(len(multi_obs))]).flatten()

        # Combine the predictions using the weights for the current row
        prediction = np.dot(w, o_preds)
        predictions.append(prediction)

    return np.array(predictions)


def check_observers_and_wandb_upload(tot_true, tot_pred, conf, output_dir, comparison_3d=True):
    """
    Check observers and optionally upload results to wandb.
    """
    # run_wandb = conf.experiment.run_wandb
    n_obs = conf.model_parameters.n_obs

    for el in range(n_obs):
        label = f"obs_{el}"

        # tot_obs_pred = np.vstack((tot_pred[0], tot_pred[1], pred)).T
        # run_figs = os.path.join(output_dir, label)

        # if run_wandb:
        #     aa = OmegaConf.load(f"{run_figs}/config.yaml")
        #     print(f"Initializing wandb for observer {el}...")
        #     wandb.init(project=name, name=label, config=aa)
                
        pp.plot_validation_3d(tot_true[:, 0:2], tot_true[:, -1], tot_pred[:, -1], output_dir)
        observers = {
        "grid": tot_pred[:, :2],
        "theta": tot_pred[:, -1],
        "label": "observers",
        }

        observers_gt = {
            "grid": tot_true[:, :2],
            "theta": tot_true[:, -1],
            "label": "observers_gt",
        }

        system_gt = {
            "grid": tot_true[:, :2],
            "theta": tot_true[:, 2],
            "label": "system_gt",
        }
        run_figs = output_dir
        # os.makedirs(run_figs, exist_ok=True)
        pp.plot_multiple_series([observers, observers_gt, system_gt], run_figs)
        pp.plot_l2(system_gt, [observers, observers_gt], run_figs)
        matching = extract_matching(tot_true, tot_pred)
        metrics = compute_metrics(matching[:, 0:2], matching[:, 3+el], matching[:, 3+n_obs+1+el], run_figs, system=matching[:, 2])
        # struttura di matching: x, t, sys_matlab, obs_matlab, mm_obs_matlab, obs_pinns, mm_obs_pinns
        if comparison_3d:
            pp.plot_comparison_3d(tot_true[:, 0:2], tot_true[:, 2], tot_pred[:, -1], run_figs)

        # if run_wandb:
        #     wandb.log(metrics)
        #     wandb.finish()



def check_system_and_wandb_upload(tot_true, tot_pred, conf, run_figs, comparison_3d=True):
    """
    Check observers and optionally upload results to wandb.
    """
    # run_wandb = conf.experiment.run_wandb
    # name = conf.experiment.name
    # if run_wandb:
    #     aa = OmegaConf.load(f"{run_figs}/config.yaml")
    #     print(f"Initializing wandb for system...")
    #     wandb.init(project=name, name=f"system", config=aa)
            
    system = {
        "grid": tot_pred[:, :2],
        "theta": tot_pred[:, -1],
        "label": "system",
    }

    system_gt = {
        "grid": tot_true[:, :2],
        "theta": tot_true[:, 2],
        "label": "system_gt",
    }
    pp.plot_multiple_series([system, system_gt], run_figs)
    pp.plot_l2(system, [system_gt], run_figs)

    if comparison_3d:
        matching = extract_matching(tot_true, tot_pred)
        pp.plot_validation_3d(tot_true[:, 0:2], tot_true[:, 2], tot_pred[:, -1], run_figs, system=True)

    metrics = compute_metrics(matching[:, 0:2], matching[:, 2], matching[:, 3], run_figs)
    # if run_wandb:
    #     matching = extract_matching(tot_true, tot_pred)
    #     wandb.log(metrics)
    #     wandb.finish()

    return metrics


def get_system_pred(model, X, output_dir):
    preds = [X[:, 0], X[:, -1]]
    y_sys_pinns = model.predict(X)
    data_to_save = np.column_stack((X[:, 0].round(n_digits), X[:, -1].round(n_digits), y_sys_pinns.round(n_digits)))
    np.savetxt(f'{output_dir}/prediction_system.txt', data_to_save, fmt='%.2f %.2f %.4f', delimiter=' ') 

    preds = np.array(data_to_save).reshape(len(preds[0]), 3).round(n_digits)
    return preds


def get_observers_preds(multi_obs, x_obs, output_dir, conf):
    n_obs = conf.model_parameters.n_obs
    preds = [x_obs[:, 0], x_obs[:, -1]]
    if n_obs==1:
        obs_pred = multi_obs.predict(x_obs)
        obs_pred = obs_pred.reshape(len(obs_pred),)
        # run_figs = os.path.join(output_dir, f"obs_0")
        data_to_save = np.column_stack((x_obs[:, 0].round(n_digits), x_obs[:, -1].round(n_digits), obs_pred.round(n_digits)))
        np.savetxt(f'{output_dir}/prediction_obs_0.txt', data_to_save, fmt='%.2f %.2f %.4f', delimiter=' ') 
        preds.append(obs_pred)
        preds = np.array(preds).reshape(3, len(preds[0])).round(n_digits)
        OmegaConf.save(conf, f"{output_dir}/config.yaml")
        return preds.T

    else:
        for el in range(n_obs):
            obs_pred = multi_obs[el].predict(x_obs)
            run_figs = os.path.join(output_dir, f"obs_{el}")
            obs_pred = obs_pred.reshape(len(obs_pred),)
            data_to_save = np.column_stack((x_obs[:, 0].round(n_digits), x_obs[:, -1].round(n_digits), obs_pred.round(n_digits)))
            np.savetxt(f'{run_figs}/prediction_obs_{el}.txt', data_to_save, fmt='%.2f %.2f %.4f', delimiter=' ')
            preds.append(obs_pred)

        run_figs = co.set_run(f"mm_obs")
        conf.model_properties.W = None
        OmegaConf.save(conf, f"{run_figs}/config.yaml")
        mm_pred = solve_ivp(multi_obs, run_figs, conf, x_obs)
        preds.append(mm_pred)
        preds = np.array(preds).reshape(len(multi_obs)+3, len(preds[0])).round(n_digits)
        return preds.T


def get_scaled_labels(rescale):
    xlabel=r"$x \, (m)$" if rescale else "X"
    ylabel=r"$t \, (s)$" if rescale else r"$\tau$"
    zlabel=r"$T \, (^{\circ}C)$" if rescale else r"$\theta$"
    return xlabel, ylabel, zlabel


def get_plot_params(conf):
    """
    Load plot parameters based on configuration for each entity (system, observers, etc.),
    and set up characteristics such as colors, linestyles, linewidths, and alphas.
    
    :param conf: Configuration object loaded from YAML.
    :return: Dictionary containing plot parameters for each entity.
    """
    # exp_name = conf.experiment.name

    # Load entity-specific configurations from the config
    entities = conf.plot.entities

    # System parameters
    system_params = {
        "color": entities.system.color,
        "label": entities.system.label,
        "linestyle": entities.system.linestyle,
        "linewidth": entities.system.linewidth,
        "alpha": entities.system.alpha
    }

    theory_params = {
        "color": entities.theory.color,
        "label": entities.theory.label,
        "linestyle": entities.theory.linestyle,
        "linewidth": entities.theory.linewidth,
        "alpha": entities.theory.alpha
    }

    bound_params = {
        "color": entities.bound.color,
        "label": entities.bound.label,
        "linestyle": entities.bound.linestyle,
        "linewidth": entities.bound.linewidth,
        "alpha": entities.bound.alpha
    }

    # Multi-observer parameters
    multi_observer_params = {
        "color": entities.multi_observer.color,
        "label": entities.multi_observer.label,
        "linestyle": entities.multi_observer.linestyle,
        "linewidth": entities.multi_observer.linewidth,
        "alpha": entities.multi_observer.alpha
    }

    # Observers parameters (dynamically adjust for number of observers)
    n_obs = conf.model_parameters.n_obs

    # Conditional handling based on the number of observers
    if n_obs == 1:
        observer_params = {
            "color": entities.observers.color[0],
            "label": entities.observers.label[0],
            "linestyle": entities.observers.linestyle[0],
            "linewidth": entities.observers.linewidth[0],
            "alpha": entities.observers.alpha[0]
        }
    else:
        observer_params = {
            "color": entities.observers.color[:n_obs],
            "label": entities.observers.label[:n_obs],
            "linestyle": entities.observers.linestyle[:n_obs],
            "linewidth": entities.observers.linewidth[:n_obs],
            "alpha": entities.observers.alpha[:n_obs]
        }

    # Ground truth parameters
    system_gt_params = {
        "color": entities.system_gt.color,
        "label": entities.system_gt.label,
        "linestyle": entities.system_gt.linestyle,
        "linewidth": entities.system_gt.linewidth,
        "alpha": entities.system_gt.alpha
    }

    multi_observer_gt_params = {
        "color": entities.multi_observer_gt.color,
        "label": entities.multi_observer_gt.label,
        "linestyle": entities.multi_observer_gt.linestyle,
        "linewidth": entities.multi_observer_gt.linewidth,
        "alpha": entities.multi_observer_gt.alpha
    }

    if n_obs == 1:
        observer_gt_params = {
            "color": entities.observers_gt.color[0],
            "label": entities.observers_gt.label[0],
            "linestyle": entities.observers_gt.linestyle[0],
            "linewidth": entities.observers_gt.linewidth[0],
            "alpha": entities.observers_gt.alpha[0]
        }
    else:
        observer_gt_params = {
            "color": entities.observers_gt.color[:n_obs],
            "label": entities.observers_gt.label[:n_obs],
            "linestyle": entities.observers_gt.linestyle[:n_obs],
            "linewidth": entities.observers_gt.linewidth[:n_obs],
            "alpha": entities.observers_gt.alpha[:n_obs]
        }

    # Adjust markers if experiment name starts with "meas_"
    markers = [None] * n_obs
    # if exp_name[1].startswith("meas_"):
    #     markers[0] = "*"

    return {
        "system": system_params,
        "theory": theory_params,
        "bound": bound_params,
        "multi_observer": multi_observer_params,
        "observers": observer_params,
        "system_gt": system_gt_params,
        "multi_observer_gt": multi_observer_gt_params,
        "observers_gt": observer_gt_params,
        "markers": markers
    }

def solve_ivp(multi_obs, fold, conf, x_obs):
    """
    Solve the IVP for observer weights and plot the results.
    """
    n_obs = conf.model_parameters.n_obs
    lam = conf.model_parameters.lam
    ups = conf.model_parameters.upsilon
    p0 = np.full((n_obs,), 1/n_obs)

    def f(t, p):
        a = mu(multi_obs, t)
        e = np.exp(-1 * a)
        d = np.inner(p, e)
        f_list = []
        for el in range(len(p)):
            f_el = - lam * (1 - (e[:, el] / d)) * p[el]
            f_list.append(f_el)
        return np.array(f_list).reshape(len(f_list),)

    sol = integrate.solve_ivp(f, (0, 1), p0, t_eval=np.linspace(0, 1, 100))
    weights = np.zeros((sol.y.shape[0] + 1, sol.y.shape[1]))
    weights[0] = sol.t
    weights[1:] = sol.y
    
    np.save(f'{fold}/weights_l_{lam}_u_{ups}.npy', weights)
    pp.plot_weights(weights[1:], weights[0], fold, conf)
    y_pred = mm_predict(multi_obs, x_obs, fold)
    data_to_save = np.column_stack((x_obs[:, 0].round(n_digits), x_obs[:, -1].round(n_digits), y_pred.round(n_digits)))
    np.savetxt(f'{fold}/prediction_mm_obs.txt', data_to_save, fmt='%.2f %.2f %.4f', delimiter=' ')
    return y_pred

 
def run_matlab_ground_truth(prj_figs):
    """
    Optionally run MATLAB ground truth.
    """

    # cfg = get_configuration(prj_figs, "ground_truth")
    cfg = OmegaConf.load(f"{prj_figs}/config.yaml")

    n_obs = cfg.model_parameters.n_obs
    cfg.model_properties.W = cfg.model_parameters.W4

    print("Running MATLAB ground truth calculation...")
    eng = matlab.engine.start_matlab()
    eng.cd(f"{src_dir}/matlab", nargout=0)
    eng.BioHeat(nargout=0)
    eng.quit()

    solution = gen_testdata(cfg)
    X, y_sys, y_observers, y_mmobs = solution[:, 0:2], solution[:, 2], solution[:, 3:3+n_obs], solution[:, -1]
    t = np.unique(X[:, 1])
    metr = compute_metrics(X, y_observers, y_observers, prj_figs, system=y_sys)
    y_theory, y_bound = compute_y_theory(X, y_sys, y_observers)

    system_gt = { "grid": X, "theta": y_sys, "label": "system_gt"}
    observer_gt = {"grid": X, "theta": y_observers, "label": "observers_gt"}
    theory = {"grid": X, "theta": y_theory, "label": "theory"}
    bound = {"grid": X, "theta": y_bound, "label": "bound"}

    pp.plot_multiple_series([system_gt, observer_gt], prj_figs)
    pp.plot_l2(system_gt, [observer_gt, theory, bound], prj_figs)

    # if n_obs==1:
        # pp.plot_tf_matlab_1obs(X, y_sys, y_observers, prj_figs)
    #     pp.plot_l2_matlab_1obs(X, y_sys, y_observers, prj_figs)
    #     pp.plot_comparison_3d(X, y_sys, y_observers, prj_figs, gt= True)
        # pp.plot_generic_5_figs(tot_true=solution, tot_pred=None, number=None, prj_figs=prj_figs)

    # else:
    #     mu = compute_mu(conf1)
    #     pp.plot_mu(mu, t, prj_figs, gt=True)

    #     t, weights = load_weights(conf1)
    #     pp.plot_weights(weights, t, prj_figs, conf1, gt=True)
    #     pp.plot_tf_matlab(X, y_sys, y_observers, y_mmobs, prj_figs)
    #     pp.plot_comparison_3d(X, y_sys, y_mmobs, prj_figs, gt= True)
    #     pp.plot_l2_matlab(X, y_sys, y_observers, y_mmobs, prj_figs)

        # y1_matlab, gt1_matlab, gt2_matlab, y2_matlab = point_ground_truths(conf1)
        # df = load_from_pickle(f"{src_dir}/data/vessel/{string}.pkl")
        # pp.plot_timeseries_with_predictions(df, y1_matlab, gt1_matlab, gt2_matlab, y2_matlab, prj_figs, gt=True)

    print("MATLAB ground truth completed.")
    print("Metrics:", metr["total_L2RE_sys"])
    return metr["total_L2RE_sys"]


def compute_y_theory(grid, sys, obs):
    str = np.where(np.abs(cc.W_sys - cc.W_obs) <= 1e-08, 'exact', 'diff')
    x = np.unique(grid[:, 0])
    t = np.unique(grid[:, -1])
    sys_0 = sys[:len(x)]
    sys_0 = sys_0.reshape(len(sys_0), 1)
    obs_0 = obs[:len(x)]
    obs_0 = obs_0.reshape(len(obs_0), 1)
    l2_0 = calculate_l2(grid[grid[:, 1]==0], sys_0, obs_0)

    # decay = cc.decay_rate_exact if str=='exact' else cc.decay_rate_diff if str=='diff'
    decay = getattr(cc, f"decay_rate_{str}")

    return l2_0 * np.exp(-t*decay), np.full_like(t, cc.c_0)




def calculate_l2(e, true, pred):
    l2 = []
    true = true.reshape(len(e), 1)
    pred = pred.reshape(len(e), 1)
    tot = np.hstack((e, true, pred))
    t = np.unique(tot[:, 1])
    x = np.unique(tot[:, 0])
    delta_x = 0.01 if len(x)==1 else x[1]- x[0]
    for el in t:
        tot_el = tot[tot[:, 1] == el]
        el_true = tot_el[:, 2]
        el_pred = tot_el[:, 3]
        el_err = el_true - el_pred
        l2_el = np.sum(el_err**2)*delta_x
        # l2_el = dde.metrics.l2_relative_error(tot_el[:, 2], tot_el[:, 3])
        
        l2.append(np.sqrt(l2_el))
    return np.array(l2)


def calculate_mse(true, pred):

    true = true.reshape(len(true), 1)
    pred = pred.reshape(len(true), 1)
    err = true - pred
    mse = np.sum(err**2)/len(err)

    return np.array(mse)


def solve_ic_comp(y1_0, y2_0, y3_0, K, a5):
    # Equation 1: b_4 = y1_0
    b_4 = y1_0
    
    # Equation 2: b_3 = y2_0 - b_4
    b_3 = y2_0 - b_4
    
    # Equation 3: b_2 = b_3 + K*(b_3 + b_4) + a5*(y3_0 - y2_0) + K*y2_0
    b_2 = b_3 + K * (b_3 + b_4) + a5 * (y3_0 - y2_0) + K * y2_0
    
    return b_2, b_3, b_4


def parse_line(line):
    parts = line.strip().split(', ')
    time_str = parts[0].split()[1]  # Extract the time (HH:MM:SS.microsecond)
    time = datetime.datetime.strptime(time_str, '%H:%M:%S.%f').time()
    measurements = parts[2:]  # Skip date and 'Temperature' keyword
    data = {}
    for i in range(0, len(measurements), 2):
        point = int(measurements[i])
        temperature = float(measurements[i + 1])
        if point not in data:
            data[point] = []
        data[point].append((time, temperature))
    return data


def load_measurements(file_path):
    timeseries_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = parse_line(line)
            for point, measurements in data.items():
                if point not in timeseries_data:
                    timeseries_data[point] = []
                timeseries_data[point].extend(measurements)
    return timeseries_data


def save_to_pickle(data, file_path):
    with open(file_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_from_pickle(file_path):
    with open(file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)
    

def extract_entries(timeseries_data, tmin, tmax, threshold):
    keys_to_extract = {10: 'y1', 45: 'gt1', 66: 'gt2', 24: 'y2', 31: 'y3', 37:'bol_out'}  # original bol_out: 39
    extracted_data = {new_key: timeseries_data.get(old_key, []) for old_key, new_key in keys_to_extract.items()}

    # Create a list of all unique times
    all_times = sorted(set(time for times in extracted_data.values() for time, temp in times))
    
    # # Normalize times to seconds, starting from zero
    start_time = all_times[0]
    all_times_in_seconds = [(datetime.datetime.combine(datetime.date.today(), time) - 
                             datetime.datetime.combine(datetime.date.today(), start_time)).total_seconds() 
                            for time in all_times]
    
    # Initialize the dataframe
    df = pd.DataFrame({'t': np.array(all_times_in_seconds).round()})
    
    # Populate the dataframe with temperatures
    for key, timeseries in extracted_data.items():
        temp_dict = {time: temp for time, temp in timeseries}
        df[key] = [temp_dict.get(time, float('nan')) for time in all_times]
    
    df = df[(df['t']>tmin) & (df['t']<tmax)].reset_index(drop=True)

    # return df
    df['time_diff'] = df['t'].diff()#.dt.total_seconds()

    # Identify the indices where a new interval starts
    new_intervals = df[df['time_diff'] > threshold].index

    # Include the first index as the start of the first interval
    new_intervals = [0] + list(new_intervals)

    # Create an empty list to store the last measurements of each interval
    last_measurements = []

    # Extract the last measurement from each interval
    for i in range(len(new_intervals)):
        start_idx = new_intervals[i]
        end_idx = new_intervals[i + 1] - 1 if i + 1 < len(new_intervals) else len(df) - 1
        last_measurements.append(df.iloc[end_idx])

    # Create the df_short DataFrame from the list of last measurements
    df_short = pd.DataFrame(last_measurements).drop(columns=['time_diff']).reset_index(drop=True)

    return df_short


def scale_df(df):
    time = df['t']-df['t'][0]
    new_df = pd.DataFrame({'tau': scale_time(time)})

    for ei in ['y1', 'gt1', 'gt2', 'y2', 'y3']:
        new_df[ei] = scale_t(df[ei])    
    return new_df


def rescale_df(df):
    time = df['tau']
    new_df = pd.DataFrame({'t': rescale_time(time)})

    for ei in ['y1', 'gt1', 'gt2', 'y2', 'y3']:
        new_df[ei] = rescale_t(df[ei])    
    return new_df


def SAR(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    return cc.beta*torch.exp(-cc.cc*(x-cc.x0))*cc.SAR_0


def point_predictions(preds):
    """
    Generates and scales predictions from the multi-observer model.
    """
    positions = get_tc_positions()
    
    # Extract predictions based on positions
    y2_pred_sc = preds[preds[:, 0] == positions[0]][:, -1]
    gt2_pred_sc = preds[preds[:, 0] == positions[1]][:, -1]
    gt1_pred_sc = preds[preds[:, 0] == positions[2]][:, -1]
    y1_pred_sc = preds[preds[:, 0] == positions[3]][:, -1]

    return y1_pred_sc, gt1_pred_sc, gt2_pred_sc, y2_pred_sc


def point_ground_truths(conf):
    """
    Generates and scales predictions from the multi-observer model.
    """
    
    positions = get_tc_positions()
    X, _, _, y_mmobs = gen_testdata(conf)

    truths = np.hstack((X, y_mmobs))
    
    # Extract predictions based on positions
    y2_truth_sc = truths[truths[:, 0] == positions[0]][:, 2]
    gt2_truth_sc = truths[truths[:, 0] == positions[1]][:, 2]
    gt1_truth_sc = truths[truths[:, 0] == positions[2]][:, 2]
    y1_truth_sc = truths[truths[:, 0] == positions[3]][:, 2]

    return y1_truth_sc, gt1_truth_sc, gt2_truth_sc, y2_truth_sc


# def configure_settings(cfg, experiment):
#     cfg.model_properties.direct = False
#     cfg.model_properties.W = cfg.model_parameters.W4
#     exp_type_settings = getattr(cfg.experiment.type, experiment[0])
#     cfg.model_properties.pwr_fact=exp_type_settings["pwr_fact"]
#     cfg.model_properties.h=exp_type_settings["h"]

#     if experiment[1].startswith("simulation"):
#         name_exp = "simulation"
#     else:
#         name_exp = experiment[1]
        
#     meas_settings = getattr(exp_type_settings, name_exp)
#     cfg.model_properties.Ty10=meas_settings["y1_0"]
#     cfg.model_properties.Ty20=meas_settings["y2_0"]
#     cfg.model_properties.Ty30=meas_settings["y3_0"]
#     cfg.model_parameters.gt1_0=meas_settings["gt1_0"]
#     cfg.model_parameters.gt2_0=meas_settings["gt2_0"]

#     return cfg


def extract_matching(tot_true, tot_pred):
    # Extract the columns from tot_true
    
    xs = np.unique(tot_true[:, 0])
    filtered_true=[]
    tot_true[:, 1] = tot_true[:, 1].round(n_digits)
    for el in np.unique(tot_true[:, 1]):
        for i in range(len(xs)):
            el_pred = tot_true[tot_true[:, 1]==el][i]
            filtered_true.append(el_pred)

    tot_true = np.array(filtered_true)

    match = []
    for el in np.unique(tot_true[:, 0]):
        trues = tot_true[tot_true[:, 0]==el]
        preds = tot_pred[tot_pred[:, 0]==el][:, 2:]
        match_el = np.hstack((trues, preds))
        for tt in range(len(match_el)):
            match.append(match_el[tt])

    # Convert new_data to a numpy array
    return np.array(match)

def initialize_run(cfg1):
    rel_out_dir = cfg1.run.dir
    abso = os.path.dirname(git_dir)
    output_dir = f"{abso}/{rel_out_dir[2:]}"

    os.makedirs(output_dir, exist_ok=True)
    cfg1.output_dir = output_dir
    OmegaConf.save(cfg1,f"{output_dir}/config.yaml")
    OmegaConf.save(cfg1,f"{conf_dir}/config_run.yaml")

    return cfg1, output_dir