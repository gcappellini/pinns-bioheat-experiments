import deepxde as dde
import numpy as np
from numpy.linalg import norm
import os, logging
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
# import matlab.engine
from hydra import initialize, compose
from common import setup_log


dde.config.set_random_seed(200)
np.random.seed(200)
torch.manual_seed(200)
dde.config.set_default_float("float64")

dev = "cuda" if torch.cuda.is_available() else "cpu"

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
conf_dir = os.path.join(src_dir, "configs")
models = os.path.join(git_dir, "models")
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(models, exist_ok=True)

f1, f2, f3 = [None]*3
n_digits = 6


def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(0)
    return round(losshistory.loss_train[0], 3)


def compute_metrics(series_to_plot, cfg, run_figs):
    # Load loss weights from configuration
    matching = extract_matching(series_to_plot)
    props = cfg.model_properties
    loss_weights = [props.w_res, props.w_bc0, props.w_bc1, props.w_ic]
    small_number = 1e-8
    
    grid = matching[:, :2]
    true = matching[:, 2]
    true_nonzero = np.where(true != 0, true, small_number)
    
    metrics = {}
    parts = []

    # Iterate over each part in series_to_plot[1:]

    for i in range(1, len(series_to_plot)):
        part_name = series_to_plot[i]["label"]
        parts.append(part_name)
        pred = matching[:, 2 + i]
        pred_nonzero = np.where(pred != 0, pred, small_number)
        
        # Part 1: General metrics for pred (Observer PINNs vs Observer MATLAB)
        L2RE = np.sum(series_to_plot[i]["L2_err"])
        MSE = calculate_mse(true, pred)
        max_err = np.max(np.abs(true_nonzero - pred))
        mean_err = np.mean(np.abs(true_nonzero - pred))
        
        metrics.update({
            f"{part_name}_L2RE": L2RE,
            f"{part_name}_MSE": MSE,
            f"{part_name}_max": max_err,
            f"{part_name}_mean": mean_err,
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
            true_cond = true[condition]
            pred_cond = pred[condition]
            true_nonzero_cond = np.where(true_cond != 0, true_cond, small_number)
            
            # Calculate metrics for pred under this specific condition

            L2RE_cond = 0
            MSE_cond = calculate_mse(true_cond, pred_cond)
            max_err_cond = np.max(np.abs(true_nonzero_cond - pred_cond))
            mean_err_cond = np.mean(np.abs(true_nonzero_cond - pred_cond))
            
            # Store these metrics in the dictionary
            metrics.update({
                f"{part_name}_{cond_name}_L2RE": L2RE_cond,
                f"{part_name}_{cond_name}_MSE": MSE_cond,
                f"{part_name}_{cond_name}_max": max_err_cond,
                f"{part_name}_{cond_name}_mean": mean_err_cond,
            })

        # Calculate and store LOSS metric for each condition
        LOSS_pred = np.sum(loss_weights * np.array([metrics[f"{part_name}_{cond}_MSE"] for cond in ["domain", "bc0", "bc1", "initial_condition"]]))
        metrics[f"{part_name}_LOSS"] = LOSS_pred
        for cond_name in conditions.keys():
            metrics[f"{part_name}_{cond_name}_LOSS"] = loss_weights[0] * metrics.get(f"{part_name}_{cond_name}_MSE", 0)

    # Save the metrics dictionary as a YAML file
    # Convert metrics values to float (Hydra supports float, not float64)
    metrics = {k: float(v) for k, v in metrics.items()}
    with open(f"{run_figs}/metrics.yaml", "w") as file:
        OmegaConf.save(config=OmegaConf.create(metrics), f=file)

    return metrics


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def create_X_anchor(n_ins, num_points=cc.n_anchor_points):
    if num_points == 0:
        return None
    # Create the time component ranging from 0 to 1
    time = np.linspace(0, 1, num_points)
    
    # Create the space component ranging from 0 to 0.3
    space = np.linspace(0, 0.3, num_points)

    if n_ins == 2:
        # Create the second component ranging from 0 to 0.6
        X_anchor = np.array(np.meshgrid(space, time)).T.reshape(-1, n_ins)

    elif n_ins == 3:
        # Create the second component ranging from 0 to 0.6
        y_2 = np.linspace(0, 0.6, num_points)
        X_anchor = np.array(np.meshgrid(space, y_2, time)).T.reshape(-1, n_ins)
    
    elif n_ins == 4:
        # Create the second component ranging from 0 to 0.2
        y_1 = np.linspace(0, 0.2, num_points)
        # Create the third component ranging from 0 to 0.6
        y_2 = np.linspace(0, 0.6, num_points)
        X_anchor = np.array(np.meshgrid(space, y_1, y_2, time)).T.reshape(-1, n_ins)

    
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
    inverse = config.experiment.inverse
    run = config.experiment.run

    if inverse and run.startswith("meas"):
        meas, _ = import_testdata(config)
        mask = np.isin(meas["grid"][:, 1], [0])
        x_points = meas["grid"][mask][:, 0]
        ic_meas = meas["theta"][mask]
        x_points=x_points.reshape(ic_meas.shape)
        ic_meas_interp = interp1d(x_points.flatten(), ic_meas.flatten(), kind="quadratic", fill_value="extrapolate")
        ic_meas = ic_meas_interp(x_points.flatten())

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
    b1, b2, b3, b4 = cc.b1, cc.b2, cc.b3, cc.b4
    c1, c2, c3 = cc.c1, cc.c2, cc.c3 
    K = cc.K
    # delta_x = cc.delta_x

    W = dde.Variable(cc.W_min) if inverse else model_props.W
    theta10, theta20, theta30 = cc.theta10, cc.theta20, cc.theta30
    time_index = n_ins -1


    def ic_fun(x):
        z = x if len(x.shape) == 1 else x[:, :1]

        if n_ins==2 and run.startswith("meas"):
            e = ic_meas_interp(z.cpu().detach())
            e = torch.tensor(ic_meas_interp(z.cpu().detach()), device=x.device)
            return e

        elif n_ins==2 and not run.startswith("meas"):
            return b1 * z**3 + b2 * z**2 + b3 * z + b4
        else:
            return c1 * z**2 + c2 * z + c3
        
    def bc0_fun(x, theta, _):
        
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
        
        y3 = cc.theta30 #if n_ins == 2 else x[:, 3:4] if n_ins == 5 else x[:, 2:3]
        y2 = None if n_ins == 2 else x[:, 1:2] if n_ins==3 else x[:, 2:3]
        
        flusso = a5 * (y3 - theta) if n_ins==2 else a5 * (y3 - y2)

        if n_ins == 2:
            return dtheta_x + flusso
        else:
            return dtheta_x + flusso - K * (theta - y2)


    def output_transform(x, y):
        y1 = cc.theta10 if n_ins<=3 else x[:, 1:2]
        t = x[:, time_index:]
        x1 = x[:, 0:1]
        
        return t * (x1 - 1) * y + ic_fun(x) + y1 - cc.theta10

    
    def rff_transform(inputs):

        b = torch.Tensor(cc.b).to(device=dev)
        vp = 2 * np.pi * inputs @ b.T

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

    # ic = dde.icbc.IC(geomtime, ic_fun, lambda _, on_initial: on_initial)
    bc_0 = dde.icbc.OperatorBC(geomtime, bc0_fun, boundary_0)
    

    gt_path=f"{tests_dir}/cooling_ground_truth_5e-04"
    # a, _, _ = gen_testdata(config, path=gt_path)
    # mask = np.isin(a["grid"][:, 0], [0.0, 0.14, 1.0])
    # a["grid"]=a["grid"][mask]
    # a["theta"]=a["theta"][mask]
    a, _ = import_testdata(config)

    observe_x = a["grid"]
    observe_y = dde.icbc.PointSetBC(observe_x, a["theta"], component=0)

    losses = [bc_0, observe_y] if inverse else [bc_0]
    X_anchor = observe_x if inverse else create_X_anchor(n_ins)

    # Data object
    data = dde.data.TimePDE(
        geomtime,
        lambda x, theta: pde(x, theta),
        losses,
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

    if inverse: 
        return model, W 
    else: 
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
            loss_weights=round(loss_weights, 3)
            model.compile(optimizer, lr=learning_rate, loss_weights=loss_weights)
        else:
            loss_weights=[round(el, 3) for el in loss_weights]
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
    model.restore(model_path, device=dev)  # Load the weights from the saved path
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
    setup_log(f"Model trained with {optimizer} optimizer.")
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
        setup_log("Found a pre-trained model.")
        return trained_model

    # Step 1: Train with Adam optimizer
    setup_log("Training a new model with Adam optimizer.")
    conf.model_properties.optimizer = "adam"
    config_hash = co.generate_config_hash(conf.model_properties)
    model_path_adam = os.path.join(models, f"model_{config_hash}.pt")
    model, losshistory = train_and_save_model(conf, "adam", config_hash, model_path_adam)

    if conf.model_properties.iters_lbfgs>0:
        setup_log("Continue training the model with L-BFGS optimizer.")
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
    
    props, pars = conf.model_properties, conf.model_parameters
    n_ins = props.n_ins
    n = 0 if n_ins == 2 else pars.n_obs

    dir_name = path if path is not None else conf.output_dir

    file_path = next((f"{dir_name}/{file}" for file in os.listdir(dir_name) if file.startswith("output_matlab") and file.endswith(".txt")), None)

    data = np.loadtxt(file_path)
    x, t, sys = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T
    X = np.vstack((x, t)).T
    y_sys = sys.flatten()[:, None]
    system_gt = {"grid": X, "theta": y_sys, "label": "system_gt"}

    if n == 1:
        obs_id = pars.W_index
        y_obs = data[:, 3+obs_id].T
        y_obs = y_obs.flatten()[:, None]
        y_mm_obs = y_obs
    elif n > 1:
        y_obs, mmobs = data[:, 3:3+n], data[:, -1].T
        y_mm_obs = mmobs.flatten()[:, None]

    elif n==0:
        return system_gt, None, None

    
    out = np.hstack((X, y_sys, y_obs, y_mm_obs))

    label_mm_obs_gt = f"observer_{cc.W_index}_gt" if n==1 else "multi_observer_gt"

    mm_obs_gt = { "grid": out[:, :2], "theta": out[:, -1], "label": label_mm_obs_gt}

    observers_gt = [
        {
            "grid": out[:, :2], 
            "theta": out[:, 3+i], 
            "label": f"observer_{cc.W_index}_gt" if n == 1 else f"observer_{i}_gt"
        }
        for i in range(n)
    ]

    observers_gt, mm_obs_gt = calculate_l2(system_gt, observers_gt, mm_obs_gt)
    observers_gt, mm_obs_gt = compute_obs_err(system_gt, observers_gt, mm_obs_gt)
    # if n > 1:
    #     observers_gt = load_weights(observers_gt, conf, "ground_truth", path=path)
    
    return system_gt, observers_gt, mm_obs_gt


def gen_obsdata(conf, system_gt):
    global f1, f2, f3

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


def load_weights(observers, conf, label, path=None):
    pars = conf.model_parameters
    n = pars.n_obs
    # dir_name = path if path is not None else conf.output_dir
    lamb = pars.lam
    ups = pars.upsilon

    dir_name = path if path is not None else conf.output_dir

    data = np.loadtxt(f"{dir_name}/weights_l_{lamb:.1f}_u_{ups:.1f}_{label}.txt")

    for j in range(n):
        observers[j]["weights"] = data[:, j+1].reshape(data[:, 0:1].shape)

    return observers

    
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
    # Check if X is a single float value
    if isinstance(X, (int, float)):
        return X * cc.L0

    # Iterate through each component in X and rescale if it is a list-like object
    rescaled_X = []
    for part in X:
        part = np.array(part, dtype=float)  # Ensure each part is converted into a numpy array
        rescaled_part = part * cc.L0        # Apply the scaling
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
    x_gt = (cc.x_gt)/L0
    x_gt1 = (cc.x_gt1)/L0
    x_y1 = 1.0

    return {"y2": x_y2, "gt": round(x_gt, 2), "y1": x_y1}
    # return {"y2": x_y2, "gt": round(x_gt, 2), "gt1": round(x_gt1, 2),"y1": x_y1}

def get_loss_names():
    # return ["residual", "bc0", "bc1", "ic", "test", "train"]
    return ["residual", "bc0", "test", "train"]

def import_testdata(conf):
    name = conf.experiment.run
    df = load_from_pickle(f"{src_dir}/data/vessel/{name}.pkl")

    positions_dict = get_tc_positions()
    positions = list(positions_dict.values())
    taus = df['tau'].unique()
    bolus = df[df['tau'].isin(taus)][['y3']].values.flatten()
    out_bolus = np.vstack((taus, bolus)).T

    # theta_values = df[['tau', 'y2', 'gt', 'gt1', 'y1']].values
    theta_values = df[['tau'] + list(positions_dict.keys())].values
    time_arrays = [np.column_stack((positions, [time_value] * len(positions), theta_values[i, 1:])) for i, time_value in enumerate(theta_values[:, 0])]
    vstack_array = np.vstack(time_arrays)
    # Remove rows where vstack_array[:, 0] equals positions[-2]
    # vstack_array = vstack_array[vstack_array[:, 0] != positions[-2]]

    system_meas = {"grid": vstack_array[:, :2], "theta": vstack_array[:, -1].reshape(len(vstack_array[:, -1]), 1), "label": "system_meas"}

    return system_meas, out_bolus


def import_obsdata(nam):
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
    system_meas, out_bolus = import_testdata(nam)
    g = np.hstack((system_meas["grid"], system_meas["theta"].reshape(len(system_meas["grid"]), 1)))
    instants = out_bolus[:, 0]

    # Get positions for filtering rows
    positions_dict = get_tc_positions()
    positions = list(positions_dict.values())
    rows = {pos: g[g[:, 0] == pos] for pos in [positions_dict["y1"], positions_dict["y2"]]}

    # Interpolation functions
    y1 = rows[positions_dict["y1"]][:, -1].reshape(len(instants),)
    f1 = interp1d(instants, y1, kind='previous')

    y2 = rows[positions_dict["y2"]][:, -1].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    y3 = out_bolus[:, 1].reshape(len(instants),)
    f3 = interp1d(instants, y3, kind='previous')
    unique_elements = np.unique(g[:, 1])

    # x_tc = get_tc_positions()
    full_length_grid = np.linspace(0, 1, 20)

    intern_positions_dict = {k: v for k, v in positions_dict.items() if k not in ["y1", "y2"]}
    intern_positions = list(intern_positions_dict.values())
    space_array_prediction = np.sort(np.concatenate((intern_positions, full_length_grid)))

    g_xxl = np.vstack([
        np.column_stack((space_array_prediction, np.full(len(space_array_prediction), el)))
        for el in unique_elements
    ])

    Xobs = np.vstack((g_xxl[:, 0], f1(g_xxl[:, 1]), f2(g_xxl[:, 1]), g_xxl[:, 1])).T
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
    pars = config.model_parameters

    n_obs = pars.n_obs

    if n_obs == 1:

        config.model_properties.W = cc.W_obs
        setup_log(f"Training single observer with perfusion {cc.W_obs}.")
        output_model = train_model(config)
        return output_model

    multi_obs = []
    for j in range(n_obs):
        obs = cc.obs if label.startswith("simulation") else np.linspace(cc.W_min, cc.W_max, n_obs)
        perf = obs[j]
        config.model_properties.W = float(perf)
        setup_log(f"Training observer {j} with perfusion {perf}.")
        model = train_model(config)
        multi_obs.append(model)

    return multi_obs


def mu(o, tau_in, upsilon):
    global f1, f2, f3

    # tau = np.where(tau_in < 0.9944, tau_in, 0.9944)

    #     # Mapping of inputs to feature configurations
    # input_mapping = {
    #     3: lambda: np.vstack((np.zeros_like(tau), f2(tau), tau)).T,
    #     4: lambda: np.vstack((np.zeros_like(tau), f1(tau), f2(tau), tau)).T,
    #     5: lambda: np.vstack((np.zeros_like(tau), f1(tau), f2(tau), f3(tau), tau)).T
    # }

    # # Generate and return observations based on the number of inputs
    # n_inputs = o[0].net.linears[0].in_features
    # xo = input_mapping.get(n_inputs, lambda: None)()

    # muu = [calculate_mu(el.predict(xo), f2(tau), upsilon) for el in o]

    t = np.unique(o[0]["grid"][:, 1])
    closest_index = np.argmin(np.abs(t-tau_in))
    obs_err = [obs["obs_err_0.0"][closest_index] for obs in o]
    muu = [upsilon * (err ** 2) for err in obs_err]
    # print(muu)

    return np.array(muu)


def calculate_mu(os, tr, upsilon):
    tr = tr.reshape(os.shape)
    scrt = upsilon*np.abs(os-tr)**2
    return scrt


def mm_predict(multi_obs):

    predictions = []
    obs_grid = multi_obs[0]["grid"]
    instants = np.unique(multi_obs[0]["grid"][:, 1])

    for i, row in enumerate(obs_grid):
        t = row[-1]
        index_w = np.argmin(np.abs(instants- t))

        w = np.array([obs["weights"][index_w] for obs in multi_obs]).reshape(len(multi_obs), 1)

        # Predict using the multi_obs predictors for the current row
        o_preds = np.array([obs["theta"][i] for obs in multi_obs]).flatten()

        # Combine the predictions using the weights for the current row
        prediction = np.dot(w.T, o_preds.reshape(w.shape))
        predictions.append(prediction)

    return np.array(predictions).reshape(multi_obs[0]["theta"].shape)


def compute_obs_err(system, observers_data=None, mm_obs=None):

    xref_dict = get_tc_positions()

    matching = [system]
    if observers_data:
        matching.extend(observers_data)
    if mm_obs:
        matching.append(mm_obs)

    g = extract_matching(matching)

    for x_ref in xref_dict.values():
        rows_xref = g[g[:, 0] == x_ref]
        sys_xref = rows_xref[:, 2]

        if observers_data:
            n_obs = len(observers_data)
            obs_err = np.abs(rows_xref[:, 3:3+n_obs] - sys_xref[:, None])
            for i, observer in enumerate(observers_data):
                observer[f'obs_err_{x_ref}'] = obs_err[:, i]

        if mm_obs:
            mm_obs_err = np.abs(rows_xref[:, -1] - sys_xref)
            mm_obs[f'obs_err_{x_ref}'] = mm_obs_err

    return observers_data, mm_obs


def get_pred(model, X, output_dir, label):

    y_sys_pinns = model.predict(X)
    data_to_save = np.column_stack((X[:, 0].round(n_digits), X[:, -1].round(n_digits), y_sys_pinns.round(n_digits)))
    np.savetxt(f'{output_dir}/prediction_{label}.txt', data_to_save, fmt='%.2f %.2f %.6f', delimiter=' ') 

    preds = np.array(data_to_save).reshape(len(X[:, 0]), 3).round(n_digits)
    preds_dict = {"grid": preds[:, :2], "theta": preds[:, 2], "label": label}
    return preds_dict



def get_observers_preds(ground_truth, multi_obs, x_obs, output_dir, conf, label):
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
    pars = conf.model_parameters
    n_obs = pars.n_obs
    preds = [x_obs[:, 0], x_obs[:, -1]]

    # Process for multiple observers
    if isinstance(multi_obs, list):
        for el in range(n_obs):
            obs_pred = multi_obs[el].predict(x_obs).reshape(-1)
            preds.append(obs_pred)
    else:
        obs_pred = multi_obs.predict(x_obs).reshape(-1)
        preds.append(obs_pred)

    preds=np.array(preds).T
    # Prepare observer dictionaries
    obs_dict = [
        {
            "grid": preds[:, :2],
            "theta": preds[:, 2 + i],
            "label": f"observer_{i if n_obs > 1 else pars.W_index}"
        }
        for i in range(n_obs) 
    ]

    mm_obs = None

    obs_dict, mm_obs = compute_obs_err(ground_truth, obs_dict, mm_obs)

    if n_obs==1:
        mm_obs = obs_dict[0]
    else:
        obs_dict = solve_ivp(obs_dict, output_dir, conf, x_obs, label)
        mm_pred = mm_predict(obs_dict)

        mm_obs = {
            "grid": preds[:, :2],
            "theta": mm_pred,
            "label": "multi_observer"
        }

    _, mm_obs = compute_obs_err(ground_truth, obs_dict, mm_obs)
    obs_dict, mm_obs = calculate_l2(ground_truth, obs_dict, mm_obs)
    
    OmegaConf.save(conf, f"{output_dir}/config_{label}.yaml")
    for obs in obs_dict:
        lal = obs["label"]
        data_to_save = np.column_stack((obs["grid"][:, 0].round(n_digits), obs["grid"][:, -1].round(n_digits), obs["theta"].round(n_digits)))
        np.savetxt(f'{output_dir}/{lal}_{label}.txt', data_to_save, fmt='%.6f %.6f %.6f', delimiter=' ')

    data_to_save = np.column_stack((mm_obs["grid"][:, 0].round(n_digits), mm_obs["grid"][:, -1].round(n_digits), mm_obs["theta"].round(n_digits)))
    np.savetxt(f'{output_dir}/{mm_obs["label"]}_{label}.txt', data_to_save, fmt='%.6f %.6f %.6f', delimiter=' ')
    return obs_dict, mm_obs


def load_observers_preds(output_dir, conf, label):
    """
    Load predictions for observers and multi-observer models from text files.

    Args:
        output_dir: Directory where predictions are saved.
        conf: Configuration object.
        label: Label used for the prediction files.

    Returns:
        obs_dict: List of dictionaries for each observer's predictions.
        mm_obs: Dictionary for multi-observer's predictions.
    """
    pars = conf.model_parameters
    n_obs = pars.n_obs

    obs_dict = []
    for i in range(n_obs):
        file_path = os.path.join(output_dir, f"observer_{i}_{label}.txt")
        data = np.loadtxt(file_path)
        obs_dict.append({
            "grid": data[:, :2],
            "theta": data[:, 2],
            "label": f"observer_{i}"
        })

    mm_obs_file_path = os.path.join(output_dir, f"multi_observer_{label}.txt")
    mm_obs_data = np.loadtxt(mm_obs_file_path)
    mm_obs = {
        "grid": mm_obs_data[:, :2],
        "theta": mm_obs_data[:, 2],
        "label": "multi_observer"
    }

    return obs_dict, mm_obs

def get_scaled_labels(rescale):
    xlabel=r"$x \, (m)$" if rescale else "X"
    ylabel=r"$t \, (s)$" if rescale else r"$\tau$"
    zlabel=r"$T \, (^{\circ}C)$" if rescale else r"$\theta$"
    return xlabel, ylabel, zlabel


def create_params(entity, default_marker=None):
    return {
        "color": entity.color,
        "label": entity.label,
        "linestyle": entity.linestyle,
        "linewidth": entity.linewidth,
        "alpha": entity.alpha,
        "marker": getattr(entity, "marker", default_marker)
    }


def get_plot_params(conf):
    """
    Load plot parameters based on configuration for each entity (system, observers, etc.),
    and set up characteristics such as colors, linestyles, linewidths, and alphas.
    
    :param conf: Configuration object loaded from YAML.
    :return: Dictionary containing plot parameters for each entity.
    """

    entities = conf.plot.entities

    # System parameters
    system_params = create_params(entities.system)
    theory_params = create_params(entities.theory)
    bound_params = create_params(entities.bound)
    multi_observer_params = create_params(entities.multi_observer)
    train_loss_params = create_params(entities.train_loss)
    test_loss_params = create_params(entities.test_loss)
    system_gt_params = create_params(entities.system_gt)
    system_meas_params = create_params(entities.system_meas, default_marker=None)
    multi_observer_gt_params = create_params(entities.multi_observer_gt)

    # Observers parameters (dynamically adjust for number of observers)
    n_obs = conf.model_parameters.n_obs
    pos = get_tc_positions()
    observer_params = {}
    observer_gt_params = {}
    meas_points_params = getattr(entities, "meas_points")
    losses_params = getattr(entities, "losses")

    if 1<=n_obs<=8:
        for j in range(n_obs):
            i = cc.W_index if n_obs == 1 else j
            observer_params[f"observer_{i}"] = {
                "color": entities.observers.color[i],
                "label": entities.observers.label[i],
                "linestyle": entities.observers.linestyle[i],
                "linewidth": entities.observers.linewidth[i],
                "alpha": entities.observers.alpha[i],
                "marker": None
            }

            observer_gt_params[f"observer_{i}_gt"] = {
                "color": entities.observers_gt.color[i],
                "label": entities.observers_gt.label[i],
                "linestyle": entities.observers_gt.linestyle[i],
                "linewidth": entities.observers_gt.linewidth[i],
                "alpha": entities.observers_gt.alpha[i],
                "marker": None
            }

    # Return combined parameters
    return {
        "system": system_params,
        "theory": theory_params,
        "bound": bound_params,
        "multi_observer": multi_observer_params,
        "system_gt": system_gt_params,
        "system_meas": system_meas_params,
        "multi_observer_gt": multi_observer_gt_params,
        "train_loss": train_loss_params,
        "test_loss": test_loss_params,
        **observer_params,
        **observer_gt_params,
        **meas_points_params,
        **losses_params
    }

def solve_ivp(multi_obs: list, fold: str, conf: dict, x_obs, label: str):
    """
    Solve the IVP for observer weights and plot the results.
    """
    setup_log("Solving the IVP for observer weights...")
    pars = conf.model_parameters
    n_obs = pars.n_obs
    lam = pars.lam
    ups = pars.upsilon

    p0 = np.full((n_obs,), 1/n_obs)

    # t_eval = np.linspace(0, 1, 100)
    t_eval = np.unique(multi_obs[0]["grid"][:, 1])

    def f(t, p):
        a = mu(multi_obs, t, ups)
        e = np.exp(-a)

        weighted_sum = np.sum(p * e) 

        return -lam * (1 - (e / weighted_sum)) * p
    
    sol = integrate.solve_ivp(f, (0, 1), p0, t_eval=t_eval)
    weights = np.zeros((sol.y.shape[0] + 1, sol.y.shape[1]))
    weights[0] = sol.t
    weights[1:] = sol.y
    weights = weights.T
    
    np.savetxt(f"{fold}/weights_l_{lam:.1f}_u_{ups:.1f}_{label}.txt", weights, delimiter=' ')
    
    for j in range(len(multi_obs)):
        multi_obs[j]["weights"] = weights[:, j+1].reshape(weights[:, 0:1].shape)

    setup_log("IVP for observer weights solved.")
    return multi_obs

 
def run_matlab_ground_truth():
    """
    Optionally run MATLAB ground truth.
    """
    setup_log("Running MATLAB ground truth calculation...")
    print("Running MATLAB ground truth calculation...")
    eng = matlab.engine.start_matlab()
    eng.cd(f"{src_dir}/matlab", nargout=0)
    eng.BioHeat(nargout=0)
    eng.quit()
    setup_log("MATLAB ground truth calculation completed.")
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


def calculate_l2(system: dict, observers: list, mm_obs: dict):

    matching = [system, *observers, mm_obs]
    g = extract_matching(matching)
    e = g[:, 0:2]
    true = g[:, 2].reshape(len(e), 1)
    obs_pred = g[:, 3:-1]
    mm_obs_pred = g[:, -1].reshape(len(e), 1)
    t = np.unique(g[:, 1])

    for i, observer in enumerate(observers):
        pred = obs_pred[:, i]
        l2 = []
        pred = pred.reshape(len(e), 1)
        tot = np.hstack((e, true, pred))
        

        for el in t:
            tot_el = tot[tot[:, 1] == el]
            el_err = norm(np.abs(tot_el[:, 2] - tot_el[:, 3]))
            l2.append(el_err)
        observer['L2_err']=np.array(l2)

    l2 = []
    tot = np.hstack((e, true, mm_obs_pred))
    for el in t:
        tot_el = tot[tot[:, 1] == el]
        el_err = norm(np.abs(tot_el[:, 2] - tot_el[:, 3]))
        l2.append(el_err)
    mm_obs['L2_err']=np.array(l2)

    return observers, mm_obs


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
    

def extract_entries(timeseries_data, tmin, tmax, keys_to_extract={10:'y1', 45:'gt1', 66:'gt', 24:'y2', 31:'y3', 37:'bol_out'}, threshold=0.0):
 # original bol_out: 39
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

def _extract_entries(timeseries_data: dict, tmin: float, tmax: float, keys_to_extract: dict = {10: 'y1', 45: 'gt1', 66: 'gt', 24: 'y2', 31: 'y3', 37: 'bol_out'}, threshold=0.0):
    # Implemented without pandas
    extracted_data = {new_key: timeseries_data.get(old_key, []) for old_key, new_key in keys_to_extract.items()}

    # Create a list of all unique times
    all_times = sorted(set(time for times in extracted_data.values() for time, temp in times))

    # Normalize times to seconds, starting from zero
    start_time = all_times[0]
    all_times_in_seconds = [(datetime.datetime.combine(datetime.date.today(), time) -
                             datetime.datetime.combine(datetime.date.today(), start_time)).total_seconds()
                            for time in all_times]

    # Initialize the dictionary
    data_dict = {'t': np.array(all_times_in_seconds).round()}

    # Populate the dictionary with temperatures
    for key, timeseries in extracted_data.items():
        temp_dict = {time: temp for time, temp in timeseries}
        data_dict[key] = [temp_dict.get(time, float('nan')) for time in all_times]

    # Filter the data based on tmin and tmax
    filtered_indices = [i for i, t in enumerate(data_dict['t']) if tmin < t < tmax]
    filtered_data = {key: [values[i] for i in filtered_indices] for key, values in data_dict.items()}

    # Calculate time differences
    time_diff = np.diff(filtered_data['t'], prepend=filtered_data['t'][0])

    # Identify the indices where a new interval starts
    new_intervals = [i for i, diff in enumerate(time_diff) if diff > threshold]

    # Include the first index as the start of the first interval
    new_intervals = [0] + new_intervals

    # Create a list to store the last measurements of each interval
    last_measurements = []

    # Extract the last measurement from each interval
    for i in range(len(new_intervals)):
        start_idx = new_intervals[i]
        end_idx = new_intervals[i + 1] - 1 if i + 1 < len(new_intervals) else len(filtered_data['t']) - 1
        last_measurements.append({key: filtered_data[key][end_idx] for key in filtered_data})

    return last_measurements

def scale_df(df):
    time = df['t']-df['t'][0]
    new_df = pd.DataFrame({'tau': scale_time(time)})

    for ei in ['y1', 'gt1', 'gt', 'y2', 'y3']:
        new_df[ei] = scale_t(df[ei])    
    return new_df


def rescale_df(df):
    time = df['tau']
    new_df = pd.DataFrame({'t': rescale_time(time)})

    for ei in ['y1', 'gt1', 'gt', 'y2', 'y3']:
        new_df[ei] = rescale_t(df[ei])    
    return new_df


def point_predictions(pred_dicts):
    """
    Generates and scales predictions from the multi-observer model.
    """
    positions_dict = get_tc_positions()
    
    # Use extract_matching to get the combined array
    preds = extract_matching(pred_dicts)
    
    # Extract predictions based on positions
    pred_sc = []
    for entry in positions_dict.keys():
        pos=positions_dict[entry]
        dict_pred = {
            "tau": preds[preds[:, 0] == pos][:, 1],
            "theta": preds[preds[:, 0] == pos][:, -1],
            "label": entry
        }
        pred_sc.append(dict_pred)
    
    return pred_sc


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
    cfg.model_parameters.gt_0=meas_settings["gt_0"]

    return cfg


def extract_matching(dicts):
    if not dicts:
        return np.array([])

    flag = len(dicts[1]['grid']) - len(dicts[0]['grid'])

    if flag == 0:
        grid = dicts[0]['grid']
        result = np.hstack((grid, dicts[0]['theta'].reshape(-1, 1)))
        theta_obsvs = np.zeros((len(grid), len(dicts)-1))
        for i in range(1, len(dicts)):
            theta_obsvs[:, i-1] = dicts[i]['theta']
    else:
        # counter = 0
        ref_index = 1 if flag<=0 else 0
        ref = dicts[1] if flag<=0 else dicts[0]
        others = dicts
        others.pop(ref_index)
        result = np.hstack((ref["grid"], ref['theta'].reshape(-1, 1)))

        theta_obsvs = []
        for grid_elem in ref['grid']:
            distances = np.linalg.norm(others[0]["grid"] - grid_elem, axis=1)
            closest_index = np.argmin(distances)
            theta_obsvs.append([others[i]['theta'][closest_index] for i in range(len(others))])

            
            
    # Stack the matched theta values
    theta_obsvs=np.array(theta_obsvs)
    result = np.hstack((result, theta_obsvs))

    return result

