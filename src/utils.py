import deepxde as dde
import numpy as np
from numpy.linalg import norm
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
from hydra import initialize, compose


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

# If L-BFGS stops earlier than expected, set the default float type to ‘float64’:
# dde.config.set_default_float("float64")



def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(0)
    return losshistory.loss_train[0]


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


def create_X_anchor(n_ins, num_points=cc.n_anchor_points):
    # Create the time component ranging from 0 to 1
    time = np.linspace(0, 1, num_points)
    
    # Create the space component ranging from 0 to 0.3
    space = np.linspace(0, 0.3, num_points)

    if n_ins == 2:
        # Create the second component ranging from 0 to 0.6
        X_anchor = np.array(np.meshgrid(space, time)).T.reshape(-1, n_ins)

    if n_ins == 3:
        # Create the second component ranging from 0 to 0.6
        y_2 = np.linspace(0, 0.6, num_points)
        X_anchor = np.array(np.meshgrid(space, y_2, time)).T.reshape(-1, n_ins)
    
    elif n_ins == 4:
        # Create the second component ranging from 0 to 0.2
        y_1 = np.linspace(0, 0.2, num_points)
        # Create the third component ranging from 0 to 0.6
        y_2 = np.linspace(0, 0.6, num_points)
        X_anchor = np.array(np.meshgrid(space, y_1, y_2, time)).T.reshape(-1, n_ins)
    
    else:
        raise ValueError("Unsupported n_ins value. Only n_ins=2,3,4 are supported.")
    
    return X_anchor


def create_model(config):
    """
    Generalized function to create and configure a PDE solver model using DeepXDE.
    
    :param run_figs: Path to the directory containing the configuration file.
    :param direct: Boolean flag to configure the model as "sys" (direct=True) or "nbho" (direct=False).
    :param src_dir: Optional source directory for saving the configuration.
    :return: A compiled DeepXDE model.
    """
    # Load configuration
    model_props = config.model_properties

    # Extract shared parameters from the configuration
    n_ins = model_props.n_ins
    activation = model_props.activation  
    initialization = model_props.initialization
    num_dense_layers = model_props.num_dense_layers
    num_dense_nodes = model_props.num_dense_nodes
    num_domain, num_boundary, num_initial, num_test = (
        model_props.num_domain, model_props.num_boundary,
        model_props.num_initial, model_props.num_test,
    )

    a1, a2, a3, a4, a5 = cc.a1, cc.a2, cc.a3, cc.a4, cc.a5
    b1, b2, b3 = cc.b1, cc.b2, cc.b3
    K = cc.K

    W = model_props.W

    theta10, theta20, theta30 = cc.theta10, cc.theta20, cc.theta30

    time_index = n_ins -1


    def ic_fun(x):
        z = x if len(x.shape) == 1 else x[:, :1]

        if n_ins==2:
            c_2 = a5 * (theta30 - theta20)
            c_3 = theta20
            c_1 = theta10 - c_2 - c_3
            return c_1 * z**2 + c_2 * z + c_3
        
        else:
        
            return (b1 - z)*(b2 + b3 * torch.exp(K*z))


    def bc0_fun(x, theta, _):
        
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
        
        y3 = cc.theta30 #if n_ins == 2 else x[:, 3:4] if n_ins == 5 else x[:, 2:3]
        y2 = None if n_ins == 2 else x[:, 1:2] if n_ins==3 else x[:, 2:3]
        
        flusso = a5 * (y3 - theta) if n_ins==2 else a5 * (y3 - y2)

        if n_ins == 2:
            return dtheta_x + flusso
        else:
            return dtheta_x + flusso - K * (theta - y2)

    def bc1_fun(x, theta, _):
        y1 = theta10 if n_ins <=3 else x[:, 1:2]
        return theta - y1

    def bc1_hc(t):
        y1 = theta10 if n_ins <=3 else t
        return y1


    def output_transform(x, y):
        y1 = cc.theta10 if n_ins<=3 else x[:, 1:2]
        y2 = cc.theta20 if n_ins<=2 else x[:, 1:2] if n_ins==3 else x[:, 2:3]
        y3 = cc.theta30 if n_ins<=4 else x[:, 3:4]
        t = x[:, time_index:]
        x1 = x[:, 0:1]
        
        return t * (x1 - 1) * y + ic_fun(x) + bc1_hc(t)

    def h_constraint(x, t):
        # Define the hard constraint function
        hc = ic_fun(x) * (t == 0).float() + bc1_hc(t) * (x == 1).float()
        return hc
    

    def rff_transform(inputs):
        # print(inputs.shape, cc.b.shape)
        b = torch.Tensor(cc.b).to(device=dev)
        vp = 2 * np.pi * inputs @ b.T
        # print(vp.shape)
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
        
    
    def pde(x, theta):
        
        dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=time_index)
        dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
        source_term = -a3 * torch.exp(-a4 * x[:, :1])

        return a1 * dtheta_tau - dtheta_xx + W * a2 * theta + source_term


    geom_mapping = {
        2: dde.geometry.Interval(0, 1),
        3: dde.geometry.Rectangle([0, 0], [1, 1]),
        4: dde.geometry.Cuboid([0, 0, 0], [1, 0.2, 1]),
        5: dde.geometry.Hypercube([0, 0, 0, 0], [1, 0.2, 1, 1])
    }

    geom = geom_mapping.get(n_ins, None)

    timedomain = dde.geometry.TimeDomain(0, 1.5)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ic = dde.icbc.IC(geomtime, ic_fun, lambda _, on_initial: on_initial)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_fun, boundary_1)
    bc_0 = dde.icbc.OperatorBC(geomtime, bc0_fun, boundary_0)
    X_anchor = create_X_anchor(n_ins)

    # Data object
    data = dde.data.TimePDE(
        geomtime,
        lambda x, theta: pde(x, theta),
        # [bc_0, bc_1, ic],
        [bc_0],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
        anchors=X_anchor,
    )

    # Define the network
    layer_size = [n_ins] + [num_dense_nodes] * num_dense_layers + [1]
    net = dde.nn.FNN(layer_size, activation, initialization)

    net.apply_output_transform(output_transform)
    # net.apply_feature_transform(rff_transform)
    # Compile the model
    model = dde.Model(data, net)

    return model

def compile_optimizer_and_losses(model, conf):
    model_props = conf.model_properties
    initial_weights_regularizer = model_props.initial_weights_regularizer
    learning_rate = model_props.learning_rate
    # loss_weights = [model_props.w_res, model_props.w_bc0, model_props.w_bc1, model_props.w_ic]
    loss_weights = [model_props.w_res, model_props.w_bc0]
    optimizer = conf.model_properties.optimizer

    if optimizer == "adam":
        if initial_weights_regularizer:
            initial_losses = get_initial_loss(model)
            loss_weights = [
                lw * len(initial_losses) / il
                for lw, il in zip(loss_weights, initial_losses)
            ]
            model.compile(optimizer, lr=learning_rate, loss_weights=loss_weights)
        else:
            model.compile(optimizer, lr=learning_rate, loss_weights=loss_weights)
        return model

    else:
        model.compile(optimizer)

    return model



def create_callbacks(config):
    resampler = config.model_properties.resampling
    resampler_period = config.model_properties.resampler_period

    callbacks = [dde.callbacks.PDEPointResampler(period=resampler_period)] if resampler else []
    return callbacks

def restore_model(conf, model_path):
    """Restore a trained model from a file."""
    model = create_model(conf)  # Ensure the model structure is created
    model = compile_optimizer_and_losses(model, conf)
    model.restore(model_path)  # Load the weights from the saved path
    # model = torch.load(model_path, weights_only=True)
    return model


def check_for_trained_model(conf):
    """Check if a model trained with LBFGS optimizer exists."""
    conf.model_properties.optimizer = "L-BFGS"
    config_hash_lbfgs = co.generate_config_hash(conf.model_properties)
    models_files = os.listdir(models)
    # Filter for matching files
    filtered_models = [file for file in models_files if config_hash_lbfgs in file and file.endswith(".pt")]
    if not filtered_models:
        return None
    # Return the path to the first sorted model
    sorted_files = sorted(filtered_models)
    model_path = os.path.join(models, sorted_files[0])
    model = restore_model(conf, model_path)
    return model


def train_and_save_model(conf, optimizer, config_hash, save_path, pre_trained_model=None):
    """Helper function to train and save a model."""
    model = create_model(conf) if pre_trained_model is None else pre_trained_model
    conf.model_properties.optimizer = optimizer
    model = compile_optimizer_and_losses(model, conf)
    callbacks = create_callbacks(conf)

    losshistory, _ = model.train(
        iterations=conf.model_properties.iters,
        callbacks=callbacks,
        model_save_path=save_path,
        display_every=conf.plot.display_every
    )
    # Save configuration
    confi_path = os.path.join(models, f"config_{config_hash}.yaml")
    OmegaConf.save(conf, confi_path)

    return model, losshistory


def train_model(conf):
    """Train a model, checking for existing LBFGS-trained models first."""
    # Step 0: Check for LBFGS-trained model
    trained_model = check_for_trained_model(conf)

    if trained_model:
        # Return the trained model directly if found
        return trained_model

    # Step 1: Train with Adam optimizer
    conf.model_properties.optimizer = "adam"
    config_hash = co.generate_config_hash(conf.model_properties)
    model_path_adam = os.path.join(models, f"model_{config_hash}.pt")
    model, losshistory = train_and_save_model(conf, "adam", config_hash, model_path_adam)

    if conf.model_properties.iters_lbfgs>0:
        # Step 2: Train with LBFGS optimizer
        conf.model_properties.optimizer = "L-BFGS"
        config_hash = co.generate_config_hash(conf.model_properties)
        model_path_lbfgs = os.path.join(models, f"model_{config_hash}.pt")
        iters_lbfgs = conf.model_properties.iters_lbfgs
        dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=1e-08, gtol=1e-08, maxiter=iters_lbfgs, maxfun=None, maxls=50)
        model, losshistory = train_and_save_model(conf, "L-BFGS", config_hash, model_path_lbfgs, pre_trained_model=model)

    pp.plot_loss_components(np.array(losshistory.loss_train), np.array(losshistory.loss_test), np.array(losshistory.steps), config_hash)

    return model


def gen_testdata(conf, path=None):
    n = cc.n_obs
    dir_name = path if path is not None else conf.output_dir

    file_path = f"{dir_name}/output_matlab_{n}Obs.txt"

    try:
        data = np.loadtxt(file_path)
        x, t, sys = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T

        if n == 1:
            y_obs = data[:, 3:4].T
            y_obs = y_obs.flatten()[:, None]
            y_mm_obs = y_obs
        else:
            y_obs, mmobs = data[:, 3:3+n], data[:, -1].T
            y_mm_obs = mmobs.flatten()[:, None]

    except FileNotFoundError:
        print(f"File not found: {file_path}.")

    X = np.vstack((x, t)).T
    y_sys = sys.flatten()[:, None]
    
    out = np.hstack((X, y_sys, y_obs, y_mm_obs))

    label_mm_obs_gt = f"observer_{cc.W_index}_gt" if n==1 else "multi_observer_gt"

    system_gt = {"grid": out[:, :2], "theta": out[:, 2], "label": "system_gt"}
    mm_obs_gt = { "grid": out[:, :2], "theta": out[:, -1], "label": label_mm_obs_gt}

    observers_gt = [
        {
            "grid": out[:, :2], 
            "theta": out[:, 3+i], 
            "label": f"observer_{cc.W_index}_gt" if n == 1 else f"observer_{i}_gt"
        }
        for i in range(n)
    ]
    
    return system_gt, observers_gt, mm_obs_gt


def gen_obsdata(conf, path=None):
    global f1, f2, f3

    # Generate ground truth data
    system_gt, _, _ = gen_testdata(conf, path)
    n_ins = conf.model_properties.n_ins

    # Prepare grid and theta data
    g = np.hstack((system_gt["grid"], system_gt["theta"].reshape(len(system_gt["grid"]), 1)))

    # Extract unique instants
    instants = np.unique(g[:, 1])

    # Filter rows based on the first column value
    rows = {value: g[g[:, 0] == value][:, 2].reshape(len(instants),) for value in [0.0, 1.0]}

    # Define interpolation functions
    f1 = interp1d(instants, rows[1.0], kind="previous")
    f2 = interp1d(instants, rows[0.0], kind="previous")
    f3 = interp1d(instants, np.full_like(rows[0.0], cc.theta30), kind="previous")

    # Mapping of inputs to feature configurations
    input_mapping = {
        3: lambda: np.vstack((g[:, 0], f2(g[:, 1]), g[:, 1])).T,
        4: lambda: np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), g[:, 1])).T,
        5: lambda: np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    }

    # Generate and return observations based on the number of inputs
    return input_mapping.get(n_ins, lambda: None)()


def load_weights(conf, label, path=None):
    n = cc.n_obs
    dir_name = path if path is not None else conf.output_dir
    lamb = cc.lamb
    ups = cc.upsilon

    data = np.loadtxt(f"{dir_name}/{label}/weights_l_{lamb:.3f}_u_{ups:.3f}.txt")
    t, weights = data[:, 0:1], data[:, 1:1+n]

    return t, np.array(weights)

    
def scale_t(t):
    Troom = cc.Troom
    Tmax = cc.Tmax
    k = (t - Troom) / (Tmax - Troom)

    return round(k, n_digits)


def rescale_t(theta):

    Troom = cc.Troom
    Tmax = cc.Tmax

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

    # Iterate through each component in X and rescale if it is a list-like object
    rescaled_X = []
    for part in X:
        part = np.array(part, dtype=float)  # Ensure each part is converted into a numpy array
        rescaled_part = part * cc.L0           # Apply the scaling
        rescaled_X.append(rescaled_part)    # Append rescaled part to the result list

    return rescaled_X


def rescale_time(tau):
    tau = np.array(tau)
    tauf = cc.tauf
    j = tau*tauf

    return np.round(j, 0)


def scale_time(t):
    tauf = cc.tauf
    j = t/tauf

    return np.round(j, n_digits)


def get_tc_positions():
    L0 = cc.L0
    x_y2 = 0.0
    x_gt2 = (cc.x_gt2)/L0
    x_gt1 = (cc.x_gt1)/L0
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


def import_obsdata(nam, extended=True):
    """
    Import observation data and optionally generate an extended dataset.

    Args:
        nam: The name of the dataset to import.
        extended (bool): Whether to generate an extended dataset or use the original.

    Returns:
        np.ndarray: Observation data.
    """
    global f1, f2, f3

    # Import test data
    g = import_testdata(nam)
    instants = np.unique(g[:, 1])

    # Get positions for filtering rows
    positions = get_tc_positions()
    rows = {pos: g[g[:, 0] == pos] for pos in [positions[0], positions[-1]]}

    # Interpolation functions
    y2 = rows[positions[0]][:, -2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind="previous")

    # Prepare extended or regular dataset
    if extended:
        # Create grid for extended data
        x = np.linspace(0, 1, 101)
        t = np.linspace(0, 1, 101)
        X, T = np.meshgrid(x, t)
        T_clipped = np.clip(T, None, 0.9956)

        # Build extended dataset
        Xobs = np.vstack((np.ravel(X), f2(np.ravel(T_clipped)), np.ravel(T))).T
    else:
        # Build regular dataset
        Xobs = np.vstack((g[:, 0], f2(g[:, 1]), g[:, 1])).T

    return Xobs


def execute(config, label):
    """
    Executes the simulation based on the configuration and label.

    Args:
        config: Configuration object for the simulation.
        label: Simulation label (e.g., "simulation_direct" or others).

    Returns:
        Model output or list of models based on the simulation type.
    """
    # Define output directories
    out_dir = config.output_dir
    simul_dir = os.path.join(out_dir, label)

    # if label == "simulation_direct":
    #     output_model = train_model(config)
    #     return output_model

    n_obs, obs = cc.n_obs, cc.obs

    if n_obs == 1:
        W_index = config.model_parameters.W_index
        config.model_properties.W = cc.W_obs
        co.set_run(simul_dir, config, f"obs_{W_index}")
        output_model = train_model(config)
        return output_model

    multi_obs = []
    for j in range(n_obs):
        perf = obs[j]
        config.model_properties.W = float(perf)
        co.set_run(simul_dir, config, f"obs_{j}")
        model = train_model(config)
        multi_obs.append(model)

    return multi_obs


def mu(o, tau_in, upsilon):
    global f1, f2, f3

    tau = np.where(tau_in<0.9944, tau_in, 0.9944)
    xo = np.vstack((np.zeros_like(tau), f2(tau), tau)).T
    muu = []

    for el in o:
        oss = el.predict(xo)
        true = f2(tau)
        scrt = calculate_mu(oss, true, upsilon)
        muu.append(scrt)

    muu = np.column_stack(muu)


    return muu


def calculate_mu(os, tr, upsilon):
    tr = tr.reshape(os.shape)
    scrt = upsilon*np.abs(os-tr)**2
    return scrt


def compute_mu(g):
    n_obs = cc.n_obs
    upsilon = cc.upsilon

    rows_0 = g[g[:, 0] == 0.0]
    sys_0 = rows_0[:, 2:3]
    obss_0 = rows_0[:, 3:3+n_obs]

    muu = []

    for el in range(obss_0.shape[1]):
        mu_value = calculate_mu(obss_0[:, el], sys_0, upsilon)

        muu.append(mu_value)
    muu = np.column_stack(muu)#.reshape(len(muu),)
    return muu



def mm_predict(multi_obs, obs_grid, folder):

    ups = cc.upsilon
    lam = cc.lamb
    a = np.loadtxt(f'{folder}/weights_l_{lam:.3f}_u_{ups:.3f}.txt')
    weights = a[1:]

    num_time_steps = weights.shape[1]

    predictions = []

    for row in obs_grid:
        t = row[-1]
        closest_idx = int(np.round(t * (num_time_steps - 1)))
        w = weights[closest_idx]

        # Predict using the multi_obs predictors for the current row
        o_preds = np.array([multi_obs[i].predict(row.reshape(1, -1)) for i in range(len(multi_obs))]).flatten()

        # Combine the predictions using the weights for the current row
        prediction = np.dot(w[1:], o_preds)
        predictions.append(prediction)

    return np.array(predictions)


def plot_observer_results(mu, t, weights, output_dir, suffix=None):
    observers_mu = [
        {"t": t, "weight": weights[:, i], "mu": mu[:, i], "label": f"observer_{i}{suffix}"}
        for i in range(cc.n_obs)
    ]
    pp.plot_mu(observers_mu, output_dir)
    pp.plot_weights(observers_mu, output_dir)

def plot_and_compute_metrics(system_gt, series_to_plot, matching_args, conf, output_dir, system_metrics=False, comparison_3d=True):
    """
    Helper function for plotting and computing metrics.
    """
    n_ins, n_obs = conf.model_properties.n_ins, conf.model_parameters.n_obs

    # Plot general series and L2 errors
    pp.plot_multiple_series(series_to_plot, output_dir)
    pp.plot_l2(system_gt, series_to_plot[1:], output_dir)

    # Extract matching data
    matching = extract_matching(*matching_args)

    # 3D comparison plots
    if comparison_3d:
        pp.plot_validation_3d(matching[:, :2], matching[:, 2], matching[:, -1], output_dir, system=True)

    # Ground truth plots
    if output_dir.endswith("ground_truth") and n_obs>1:
        label_run = "ground_truth"
        mu = compute_mu(matching)
        t, weights = load_weights(conf, label_run)
        plot_observer_results(mu, t, weights, output_dir, suffix="_gt")

    # Multi-observer simulation plots
    if output_dir.endswith("simulation_mm_obs") and n_obs > 1:
        label_run = "simulation_mm_obs"
        mu = compute_mu(matching)[1:]
        t, weights = load_weights(conf, label_run)
        plot_observer_results(mu, t, weights, output_dir)

    # Compute and return metrics
    metrics = compute_metrics(
        matching[:, :2],
        matching[:, 2] if not system_metrics else matching[:, -1],
        matching[:, -1],
        output_dir,
        system=matching[:, 2] if system_metrics else None,
    )

    return matching, metrics


def check_and_wandb_upload(
    mm_obs_gt=None,
    mm_obs=None,
    system=None,
    system_gt=None,
    conf=None,
    output_dir=None,
    observers_gt=None,
    observers=None,
):
    """
    Check observers and optionally upload results to wandb.
    """
    if conf is None:
        raise ValueError("Configuration (`conf`) is required.")

    n_ins = conf.model_properties.n_ins if conf and conf.model_properties else False
    n_obs = conf.model_parameters.n_obs if conf and conf.model_parameters else 0
    show_obs = conf.plot.show_obs if conf and conf.plot else False

    if output_dir is None:
        raise ValueError("Output directory (`output_dir`) is required.")

    if n_ins==2 and system is not None:
        series_to_plot_direct = [system_gt, system]
        _, metrics_direct = plot_and_compute_metrics(system_gt, series_to_plot_direct, (system_gt, system), conf, output_dir, system_metrics=True)
        return metrics_direct
    
    else:
        # Indirect modeling path
        if n_obs == 1:
            if observers_gt and not observers:
                series_to_plot_n1 = [system_gt, *observers_gt]
                matching_args_n1 = (system_gt, *observers_gt)
            else:
                series_to_plot_n1 = [system_gt, *observers_gt, *observers]
                matching_args_n1 = (system_gt, *observers, mm_obs)

            _, metrics = plot_and_compute_metrics(system_gt, series_to_plot_n1, matching_args_n1, conf, output_dir, system_metrics=True)
            return metrics

        elif n_obs > 1:
            if mm_obs_gt and not mm_obs:
                series_to_plot_mm_obs = (
                    [system_gt, mm_obs_gt, *observers_gt]
                    if show_obs
                    else [system_gt, mm_obs_gt]
                    )
                matching_args_mm_obs = (system_gt, *observers_gt, mm_obs_gt)
            else:
                series_to_plot_mm_obs = (
                    [system_gt, mm_obs, mm_obs_gt, *observers_gt, *observers]
                    if show_obs
                    else [system_gt, mm_obs, mm_obs_gt]
                )
                matching_args_mm_obs = (system_gt, *observers, mm_obs)

            

            # Final metrics
            _, metrics = plot_and_compute_metrics(system_gt, series_to_plot_mm_obs, matching_args_mm_obs, conf, output_dir, system_metrics=True)
            return metrics



def get_pred(model, X, output_dir, label):

    y_sys_pinns = model.predict(X)
    data_to_save = np.column_stack((X[:, 0].round(n_digits), X[:, -1].round(n_digits), y_sys_pinns.round(n_digits)))
    np.savetxt(f'{output_dir}/prediction_{label}.txt', data_to_save, fmt='%.2f %.2f %.6f', delimiter=' ') 

    preds = np.array(data_to_save).reshape(len(X[:, 0]), 3).round(n_digits)
    preds_dict = {"grid": preds[:, :2], "theta": preds[:, 2], "label": label}
    return preds_dict



def get_observers_preds(multi_obs, x_obs, output_dir, conf):
    """
    Generate predictions for observers and multi-observer models.

    Args:
        multi_obs: Model(s) for multiple observers.
        x_obs: Input data for prediction.
        output_dir: Directory to save predictions and configuration.
        conf: Configuration object.

    Returns:
        obs_dict: List of dictionaries for each observer's predictions.
        mm_obs: Dictionary for multi-observer's predictions.
    """
    n_obs = conf.model_parameters.n_obs
    preds = [x_obs[:, 0], x_obs[:, -1]]

    # Process for a single observer
    if n_obs == 1:
        obs_pred = multi_obs.predict(x_obs).reshape(-1)
        data_to_save = np.column_stack((x_obs[:, 0], x_obs[:, -1], obs_pred)).round(n_digits)
        np.savetxt(f'{output_dir}/prediction_obs_{cc.W_index}.txt', data_to_save, fmt='%.3f %.3f %.6f', delimiter=' ')

        preds.append(obs_pred)
        preds = np.array(preds).T.round(n_digits)

        OmegaConf.save(conf, f"{output_dir}/config.yaml")

        obs_dict = [{
            "grid": preds[:, :2],
            "theta": preds[:, 2],
            "label": f"observer_{cc.W_index}"
        }]
        mm_obs = obs_dict[0]
        return obs_dict, mm_obs

    # Process for multiple observers
    for el in range(n_obs):
        obs_pred = multi_obs[el].predict(x_obs).reshape(-1)
        preds.append(obs_pred)

    # Save and configure multi-observer predictions
    conf.model_properties.W = None
    run_figs, _ = co.set_run(output_dir, conf, "mm_obs")
    OmegaConf.save(conf, f"{run_figs}/config.yaml")

    mm_pred = solve_ivp(multi_obs, output_dir, conf, x_obs)
    preds.append(mm_pred)

    # Save multi-observer predictions
    preds = np.array(preds).T.round(n_digits)
    np.savetxt(f'{output_dir}/prediction_mm_obs.txt', preds, delimiter=' ')

    # Prepare observer dictionaries
    obs_dict = [
        {
            "grid": preds[:, :2],
            "theta": preds[:, 2 + i],
            "label": f"observer_{i}"
        }
        for i in range(n_obs)
    ]
    mm_obs = {
        "grid": preds[:, :2],
        "theta": preds[:, -1],
        "label": "multi_observer"
    }
    return obs_dict, mm_obs


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

    train_loss_params = {
        "color": entities.train_loss.color,
        "label": entities.train_loss.label,
        "linestyle": entities.train_loss.linestyle,
        "linewidth": entities.train_loss.linewidth,
        "alpha": entities.train_loss.alpha
    }

    test_loss_params = {
        "color": entities.test_loss.color,
        "label": entities.test_loss.label,
        "linestyle": entities.test_loss.linestyle,
        "linewidth": entities.test_loss.linewidth,
        "alpha": entities.test_loss.alpha
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

    # Observers parameters (dynamically adjust for number of observers)
    n_obs = conf.model_parameters.n_obs

    # Prepare observer(s) parameters
    observer_params = {}
    observer_gt_params = {}

    for j in range(n_obs):

        i=cc.W_index if n_obs==1 else j

        observer_params[f"observer_{i}"] = {
            "color": entities.observers.color[i],
            "label": entities.observers.label[i],
            "linestyle": entities.observers.linestyle[i],
            "linewidth": entities.observers.linewidth[i],
            "alpha": entities.observers.alpha[i]
        }

        observer_gt_params[f"observer_{i}_gt"] = {
            "color": entities.observers_gt.color[i],
            "label": entities.observers_gt.label[i],
            "linestyle": entities.observers_gt.linestyle[i],
            "linewidth": entities.observers_gt.linewidth[i],
            "alpha": entities.observers_gt.alpha[i]
        }

    # Adjust markers if experiment name starts with "meas_"
    markers = [None] * n_obs
    # if exp_name[1].startswith("meas_"):
    #     markers[0] = "*"

    # Return combined parameters
    return {
        "system": system_params,
        "theory": theory_params,
        "bound": bound_params,
        "multi_observer": multi_observer_params,
        "system_gt": system_gt_params,
        "multi_observer_gt": multi_observer_gt_params,
        "train_loss": train_loss_params,
        "test_loss": test_loss_params,
        **observer_params,
        **observer_gt_params,
        "markers": markers
    }

def solve_ivp(multi_obs, fold, conf, x_obs):
    """
    Solve the IVP for observer weights and plot the results.
    """
    n_obs = cc.n_obs
    lam = cc.lamb
    ups = cc.upsilon

    p0 = np.full((n_obs,), 1/n_obs)

    t_eval = np.linspace(0, 1, 100)

    def f(t, p):
        a = mu(multi_obs, t, ups)
        e = np.exp(-a)

        weighted_sum = np.sum(p[:, None] * e, axis=0) 
        f_matrix = -lam * (1 - (e / weighted_sum)) * p[:, None]
        return np.sum(f_matrix, axis=1)

    sol = integrate.solve_ivp(f, (0, 1), p0, t_eval=t_eval)
    weights = np.zeros((sol.y.shape[0] + 1, sol.y.shape[1]))
    weights[0] = sol.t
    weights[1:] = sol.y
    weights = weights.T
    
    np.savetxt(f"{fold}/weights_l_{lam:.3f}_u_{ups:.3f}.txt", weights.round(n_digits), delimiter=' ')
    # pp.plot_weights(weights[1:], weights[0], fold, conf)
    y_pred = mm_predict(multi_obs, x_obs, fold)

    return y_pred

 
def run_matlab_ground_truth():
    """
    Optionally run MATLAB ground truth.
    """

    print("Running MATLAB ground truth calculation...")
    eng = matlab.engine.start_matlab()
    eng.cd(f"{src_dir}/matlab", nargout=0)
    eng.BioHeat(nargout=0)
    eng.quit()
    print("MATLAB ground truth calculation completed.")


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

    theory = {"grid": grid, "theta": l2_0 * np.exp(-t*decay), "label": "theory"}
    bound = {"grid": grid, "theta": np.full_like(t, cc.c_0), "label": "bound"}
    return theory, bound


def calculate_l2(e, true, pred):
    l2 = []
    true = true.reshape(len(e), 1)
    pred = pred.reshape(len(e), 1)
    tot = np.hstack((e, true, pred))
    t = np.unique(tot[:, 1])

    for el in t:
        tot_el = tot[tot[:, 1] == el]
        el_err = norm(np.abs(tot_el[:, 2] - tot_el[:, 3]))
        l2.append(el_err)
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


def configure_settings(cfg, experiment):
    cfg.model_properties.direct = False
    cfg.model_properties.W = cfg.model_parameters.W4
    exp_type_settings = getattr(cfg.experiment.type, experiment[0])
    cfg.model_properties.pwr_fact=exp_type_settings["pwr_fact"]
    cfg.model_properties.h=exp_type_settings["h"]

    if experiment[1].startswith("simulation"):
        name_exp = "simulation"
    else:
        name_exp = experiment[1]
        
    meas_settings = getattr(exp_type_settings, name_exp)
    cfg.model_properties.Ty10=meas_settings["y1_0"]
    cfg.model_properties.Ty20=meas_settings["y2_0"]
    cfg.model_properties.Ty30=meas_settings["y3_0"]
    cfg.model_parameters.gt1_0=meas_settings["gt1_0"]
    cfg.model_parameters.gt2_0=meas_settings["gt2_0"]

    return cfg


def extract_matching(d_true, *d_preds):
    """
    Matches true data points with predicted data points for multiple predictions.
    
    Args:
        d_true: Dictionary containing true data with keys "grid" and "theta".
        *d_preds: Multiple dictionaries containing predicted data with keys "grid" and "theta".
    
    Returns:
        A numpy array with the matched true and predicted values for all inputs.
    """
    # Extract the columns from d_true
    tot_true = np.hstack((d_true["grid"], d_true["theta"].reshape(len(d_true["grid"]), 1)))

    # Prepare the true data for matching
    xs = np.unique(tot_true[:, 0])
    filtered_true = []
    tot_true[:, 1] = tot_true[:, 1].round(n_digits)
    
    for el in np.unique(tot_true[:, 1]):
        for i in range(len(xs)):
            el_pred = tot_true[tot_true[:, 1] == el][i]
            filtered_true.append(el_pred)
    
    tot_true = np.array(filtered_true)

    # Prepare the predicted data for each d_pred
    tot_preds = []
    for d_pred in d_preds:
        pred = np.hstack((d_pred["grid"], d_pred["theta"].reshape(-1, 1)))
        tot_preds.append(pred)

    # Match true data with predicted data
    match = []
    for x in np.unique(tot_true[:, 0]):
        # Filter true values for the current x
        trues = tot_true[tot_true[:, 0] == x]
        
        # Filter predicted values for the current x for each d_pred
        preds = [pred[pred[:, 0] == x][:, 2:] for pred in tot_preds]
        
        # Combine all predicted values column-wise
        combined_preds = np.hstack(preds) if preds else np.empty((len(trues), 0))
        
        # Combine true and predicted values
        match_el = np.hstack((trues, combined_preds))
        for tt in range(len(match_el)):
            match.append(match_el[tt])
    
    # Convert match list to a numpy array
    return np.array(match)

