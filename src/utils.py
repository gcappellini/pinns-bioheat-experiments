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
# import coeff_calc as cc
import plots as pp
import common as co
from omegaconf import OmegaConf

from hydra import compose
import time

dde.config.set_default_float("float64")

dev = "cuda" if torch.cuda.is_available() else "cpu"

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
gt_dir = os.path.join(git_dir, "gt")
conf_dir = os.path.join(src_dir, "configs")
models = os.path.join(git_dir, "models")
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(models, exist_ok=True)

f1, f2, f3 = [None]*3
n_digits = 6

logger = logging.getLogger(__name__)

def setup_log(str):

    logger.info(str)


def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(0)
    return round(losshistory.loss_train[0], 3)


def compute_metrics(series_to_plot, train_info, cfg, run_figs):
    # Load loss weights from configuration
    matching = extract_matching(series_to_plot)
    # props = cfg.properties
    # loss_weights = [props.wres, props.wbc, props.w_bc1, props.w_ic]
    small_number = 1e-8
    
    grid = matching[:, :2]
    true = matching[:, 2]
    true_nonzero = np.where(true != 0, true, small_number)
    
    metrics = {}
    parts = []

    # Iterate over each part in series_to_plot[1:]
    # for i in range(1, len(series_to_plot)):
        # part_name = series_to_plot[i]["label"]
        # parts.append(part_name)
    i=1
    pred = matching[:, 2 + i]
    pred_nonzero = np.where(pred != 0, pred, small_number)
    
    # Part 1: General metrics for pred (Observer PINNs vs Observer MATLAB)
    L2RE = np.sum(series_to_plot[i]["L2_err"])
    MSE = calculate_mse(true, pred)
    max_err = np.max(np.abs(true_nonzero - pred))
    mean_err = np.mean(np.abs(true_nonzero - pred))
    
    metrics.update({
        f"L2RE": L2RE,
        f"MSE": MSE,
        f"max": max_err,
        f"mean": mean_err,
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
            f"{cond_name}_L2RE": L2RE_cond,
            f"{cond_name}_MSE": MSE_cond,
            f"{cond_name}_max": max_err_cond,
            f"{cond_name}_mean": mean_err_cond,
        })

    # # Calculate and store LOSS metric for each condition
    # LOSS_pred = np.sum(loss_weights * np.array([metrics[f"{cond}_MSE"] for cond in ["domain", "bc0", "bc1", "initial_condition"]]))
    # metrics[f"LOSS"] = LOSS_pred
    # for cond_name in conditions.keys():
    #     metrics[f"{cond_name}_LOSS"] = loss_weights[0] * metrics.get(f"{cond_name}_MSE", 0)

    # Save the metrics dictionary as a YAML file
    # Convert metrics values to float (Hydra supports float, not float64)
    # Remove metrics for conditions "bc1" and "initial_condition"
    for key in list(metrics.keys()):
        if "bc1" in key or "initial_condition" in key:
            del metrics[key]
            
    if train_info is not None:
        metrics.update({"testloss": train_info["test"].sum(axis=1).ravel().min()})

    metrics = {k: float(v) for k, v in metrics.items()}
    with open(f"{run_figs}/metrics.yaml", "w") as file:
        OmegaConf.save(config=OmegaConf.create(metrics), f=file)

    return metrics


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def create_X_anchor(nins, num_points):
    if num_points == 0:
        return None
    # Create the time component ranging from 0 to 1
    time = np.linspace(0, 1, num_points)
    
    # Create the space component ranging from 0 to 0.3
    space = np.linspace(0, 0.3, num_points)

    if nins == 2:
        # Create the second component ranging from 0 to 0.6
        X_anchor = np.array(np.meshgrid(space, time)).T.reshape(-1, nins)

    elif nins == 3:
        # Create the second component ranging from 0 to 0.6
        y_2 = np.linspace(0, 0.6, num_points)
        X_anchor = np.array(np.meshgrid(space, y_2, time)).T.reshape(-1, nins)
    
    elif nins == 4:
        # Create the second component ranging from 0 to 0.2
        y_1 = np.linspace(0, 0.2, num_points)
        # Create the third component ranging from 0 to 0.6
        y_2 = np.linspace(0, 0.6, num_points)
        X_anchor = np.array(np.meshgrid(space, y_1, y_2, time)).T.reshape(-1, nins)

    
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
    # props = config.model_properties
    pars = config.parameters
    inverse = config.experiment.inverse
    run = config.experiment.run
    hp = config.hp
    pdecoeff = config.pdecoeff

    if inverse and run.startswith("meas"):
        meas, _ = import_testdata(config)
        mask = np.isin(meas["grid"][:, 1], [0])
        x_points = meas["grid"][mask][:, 0]
        ic_meas = meas["theta"][mask]
        x_points=x_points.reshape(ic_meas.shape)
        ic_meas_interp = interp1d(x_points.flatten(), ic_meas.flatten(), kind="quadratic", fill_value="extrapolate")
        ic_meas = ic_meas_interp(x_points.flatten())

    # Extract shared parameters from the configuration
    nins = hp.nins
    af = hp.af  
    init = hp.init
    depth = hp.depth
    width = hp.width
    nres, nb, ntest = (
        hp.nres, hp.nb,
        hp.ntest,
    )

    a1, a2, a3, a4, a5 = pdecoeff.a1, pdecoeff.a2, pdecoeff.a3, pdecoeff.a4, pdecoeff.a5
    b1, b2, b3, b4 = pdecoeff.b1, pdecoeff.b2, pdecoeff.b3, pdecoeff.b4
    c1, c2, c3 = pdecoeff.c1, pdecoeff.c2, pdecoeff.c3 
    oig = pdecoeff.oig
    # delta_x = cc.delta_x

    wb = dde.Variable(pars.wbmin) if inverse else pdecoeff.wb
    y10, y20, y30 = pdecoeff.y10, pdecoeff.y20, pdecoeff.y30
    time_index = nins -1


    def ic_fun(x):
        z = x if len(x.shape) == 1 else x[:, :1]

        if nins==2 and run.startswith("meas"):
            e = ic_meas_interp(z.cpu().detach())
            e = torch.tensor(ic_meas_interp(z.cpu().detach()), device=x.device)
            return e

        elif nins==2 and not run.startswith("meas"):
            return b1 * z**3 + b2 * z**2 + b3 * z + b4
        else:
            return c1 * z**2 + c2 * z + c3
        
    def bc0_fun(x, theta, _):
        
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
        
        y3 = y30 #if nins == 2 else x[:, 3:4] if nins == 5 else x[:, 2:3]
        y2 = None if nins == 2 else x[:, 1:2] if nins==3 else x[:, 2:3]
        
        flusso = a5 * (y3 - theta) if nins==2 else a5 * (y3 - y2)

        if nins == 2:
            return dtheta_x + flusso
        else:
            return dtheta_x + flusso - oig * (theta - y2)


    def output_transform(x, y):
        y1 = y10 if nins<=3 else x[:, 1:2]
        t = x[:, time_index:]
        x1 = x[:, 0:1]
        
        return t * (x1 - 1) * y + ic_fun(x) + y1 - y10

    
    # def rff_transform(inputs):

    #     b = torch.Tensor(cc.b).to(device=dev)
    #     vp = 2 * np.pi * inputs @ b.T

    #     return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
        
    
    def pde(x, theta):
        
        dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=time_index)
        dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
        source_term = -a3 * torch.exp(-a4 * x[:, :1])

        return a1 * dtheta_tau - dtheta_xx + wb * a2 * theta + source_term


    geom_mapping = {
        2: dde.geometry.Interval(0, 1),
        3: dde.geometry.Rectangle([0, 0], [1, 1]),
        4: dde.geometry.Cuboid([0, 0, 0], [1, 0.2, 1]),
        5: dde.geometry.Hypercube([0, 0, 0, 0], [1, 0.2, 1, 1])
    }

    geom = geom_mapping.get(nins, None)

    timedomain = dde.geometry.TimeDomain(0, 1.5)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # ic = dde.icbc.IC(geomtime, ic_fun, lambda _, on_initial: on_initial)
    bc_0 = dde.icbc.OperatorBC(geomtime, bc0_fun, boundary_0)
    

    losses = [bc_0]
    X_anchor = create_X_anchor(nins, config.hp.nanc)

    if inverse:
        if run.startswith("meas"):
            a, _ = import_testdata(config)

        elif run.startswith("simulation"):
            gt_path=f"{tests_dir}/cooling_ground_truth_5e-04"
            a, _, _ = gen_testdata(config, path=gt_path)
            mask = np.isin(a["grid"][:, 0], [0.0, 0.14, 1.0])
            a["grid"]=a["grid"][mask]
            a["theta"]=a["theta"][mask]

        observe_x = a["grid"]
        observe_y = dde.icbc.PointSetBC(observe_x, a["theta"], component=0)

        losses = [bc_0, observe_y]
        X_anchor = observe_x

    # Data object
    data = dde.data.TimePDE(
        geomtime,
        lambda x, theta: pde(x, theta),
        losses,
        num_domain=nres, 
        num_boundary=nb,
        num_test=ntest,
        anchors=X_anchor,
    )

    # Define the network
    layer_size = [nins] + [width] * depth + [1]
    net = dde.nn.FNN(layer_size, af, init)

    net.apply_output_transform(output_transform)
    # net.apply_feature_transform(rff_transform)
    # Compile the model
    model = dde.Model(data, net)

    if inverse: 
        return model, wb 
    else: 
        return model

def compile_optimizer_and_losses(model, conf):
    hp = conf.hp
    iwr = hp.iwr
    lr = hp.lr
    # loss_weights = [hp.wres, hp.wbc, hp.w_bc1, hp.w_ic]
    loss_weights = [hp.wres, hp.wbc0]
    optimizer = conf.hp.optimizer

    if optimizer == "adam":
        if iwr:
            initial_losses = get_initial_loss(model)
            loss_weights = [
                lw * len(initial_losses) / il
                for lw, il in zip(loss_weights, initial_losses)
            ]
            loss_weights=round(loss_weights, 3)
            model.compile(optimizer, lr=lr, loss_weights=loss_weights)
        else:
            loss_weights=[round(el, 3) for el in loss_weights]
            model.compile(optimizer, lr=lr, loss_weights=loss_weights)
        return model

    else:
        model.compile(optimizer)

    return model


def create_callbacks(config):
    resampler = config.hp.resampling
    resampler_period = config.hp.resampler_period

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
    conf.hp.optimizer = "L-BFGS"
    config_data = OmegaConf.create({
        "pdecoeff": conf.pdecoeff,
        "hp": conf.hp
        })
    config_hash_lbfgs = co.generate_config_hash(config_data)
    models_files = os.listdir(models)
    # Filter for matching files
    filtered_models = [file for file in models_files if config_hash_lbfgs in file and file.endswith(".pt")]
    if not filtered_models:
        return None
    filtered_losses = [file for file in models_files if config_hash_lbfgs in file and file.endswith(".npz")]
    if filtered_losses:
        loss_file = os.path.join(models, filtered_losses[0])
        loss_history = np.load(loss_file)
        # info_train = {"testloss": np.min(loss_data['test']), "runtime": loss_data['runtime']}
    # else:
    #     info_train = {"testloss": None, "runtime": None}
    # Return the path to the first sorted model
    sorted_files = sorted(filtered_models)
    model_path = os.path.join(models, sorted_files[0])
    model = restore_model(conf, model_path)

    return model, loss_history


def train_and_save_model(conf, optimizer, config_hash, save_path, pre_trained_model=None):
    """Helper function to train and save a model."""
    model = create_model(conf) if pre_trained_model is None else pre_trained_model
    conf.hp.optimizer = optimizer
    model = compile_optimizer_and_losses(model, conf)
    callbacks = create_callbacks(conf)
    # start_time = time.time()

    losshistory, _ = model.train(
        iterations=conf.hp.iters,
        callbacks=callbacks,
        model_save_path=save_path,
        display_every=conf.plot.display_every
    )
    test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    # runtime = time.time() - start_time
    setup_log(f"Model trained with {optimizer} optimizer.")
    # Save configuration
    conf.testloss = round(float(test.min()), 6)
    # conf.runtime = round(float(runtime), 3)
    # conf.update({"testloss": test.min(), "runtime": runtime})
    # OmegaConf.save(conf, confi_path)
    confi_path = os.path.join(models, f"config_{config_hash}.yaml")
    np.savez(f"{save_path}_losshistory.npz", train=losshistory.loss_train, test=losshistory.loss_test, steps=losshistory.steps)
    OmegaConf.save(conf, confi_path)

    return model, losshistory


def train_model(conf):
    """Train a model, checking for existing LBFGS-trained models first."""
    # Step 0: Check for LBFGS-trained model
    trained_model = check_for_trained_model(conf)
    
    if trained_model:
        # Return the trained model directly if found
        setup_log("Found a pre-trained model.")
        model, losshistory = trained_model
        # return trained_model
    else:
        start_time=time.time()
        # Step 1: Train with Adam optimizer
        # start_time = time.time()
        setup_log("Training a new model with Adam optimizer.")
        conf.hp.optimizer = "adam"
        conf_data = OmegaConf.create({
            "hp": conf.hp,
            "pdecoeff": conf.pdecoeff
        })
        config_hash = co.generate_config_hash(conf_data)
        model_path_adam = os.path.join(models, f"model_{config_hash}.pt")
        model, losshistory = train_and_save_model(conf, "adam", config_hash, model_path_adam)

        if conf.hp.iters_lbfgs>0:
            setup_log("Continue training the model with L-BFGS optimizer.")
            # Step 2: Train with LBFGS optimizer
            conf.hp.optimizer = "L-BFGS"
            conf_data = OmegaConf.create({
                "hp": conf.hp,
                "pdecoeff": conf.pdecoeff
                })
            config_hash = co.generate_config_hash(conf_data)
            model_path_lbfgs = os.path.join(models, f"model_{config_hash}.pt")
            iters_lbfgs = conf.hp.iters_lbfgs
            dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=1e-08, gtol=1e-08, maxiter=iters_lbfgs, maxfun=None, maxls=50)
            model, losshistory = train_and_save_model(conf, "L-BFGS", config_hash, model_path_lbfgs, pre_trained_model=model)
            end_time = time.time()
            runtime=end_time-start_time
            conf.runtime=runtime
            setup_log(f"Training completed in {round(end_time-start_time, 3)} seconds.")

    conf_data = OmegaConf.create({
        "hp": conf.hp,
        "pdecoeff": conf.pdecoeff
        })
    config_hash = co.generate_config_hash(conf_data)

    pp.plot_loss_components(losshistory, config_hash, fold=conf.output_dir)
    OmegaConf.save(conf, f"{conf.output_dir}/config_{config_hash}.yaml")

    # Convert losshistory to a dictionary
    losshistory_dict = {
        "train": np.array(losshistory.loss_train),
        "test": np.array(losshistory.loss_test),
        "steps": np.array(losshistory.steps),
    }
    return model, losshistory_dict


def gen_testdata(conf):
    
    pars = conf.parameters
    nins = conf.hp.nins
    n = pars.nobs
    # conf_gt = OmegaConf.create({
    #     "pdecoeff": conf.pdecoeff,
    #     "parameters": conf.parameters})
    
    # matlab_hash = co.generate_config_hash(conf_gt)
    # path = f"{gt_dir}/gt_{matlab_hash}"
    path = conf.experiment.gt_path

    data = np.loadtxt(f"{path}.txt")
    x, t, sys = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T
    X = np.vstack((x, t)).T
    y_sys = sys.flatten()[:, None]
    system_gt = {"grid": X, "theta": y_sys, "label": "system_gt"}

    if n == 1:
        obs_id = 0 if data.shape[1]==4 else pars.obsindex
        y_obs = data[:, 3+obs_id].T
        y_obs = y_obs.flatten()[:, None]
        y_mm_obs = y_obs
    elif n > 1:
        y_obs, mmobs = data[:, 3:3+n], data[:, -1].T
        y_mm_obs = mmobs.flatten()[:, None]

    elif n==0:
        return system_gt, None, None

    
    out = np.hstack((X, y_sys, y_obs, y_mm_obs))

    label_mm_obs_gt = f"observer_{pars.obsindex}_gt" if n==1 else "multi_observer_gt"

    mm_obs_gt = { "grid": out[:, :2], "theta": out[:, -1], "label": label_mm_obs_gt}

    observers_gt = [
        {
            "grid": out[:, :2], 
            "theta": out[:, 3+i], 
            "label": f"observer_{pars.obsindex}_gt" if n == 1 else f"observer_{i}_gt"
        }
        for i in range(n)
    ]

    observers_gt, mm_obs_gt = calculate_l2(system_gt, observers_gt, mm_obs_gt)
    observers_gt, mm_obs_gt = compute_obs_err(system_gt, observers_gt, mm_obs_gt)
    if n > 1:
        observers_gt = load_weights(observers_gt, conf, path)
    
    return system_gt, observers_gt, mm_obs_gt


def gen_obsdata(conf, system_gt):
    global f1, f2, f3

    nins = conf.hp.nins

    # Prepare grid and theta data
    g = np.hstack((system_gt["grid"], system_gt["theta"].reshape(len(system_gt["grid"]), 1)))

    # Extract unique instants
    instants = np.unique(g[:, 1])

    # Filter rows based on the first column value
    rows = {value: g[g[:, 0] == value][:, 2].reshape(len(instants),) for value in [0.0, 1.0]}

    # Define interpolation functions
    f1 = interp1d(instants, rows[1.0], kind="previous")
    f2 = interp1d(instants, rows[0.0], kind="previous")
    f3 = interp1d(instants, np.full_like(rows[0.0], conf.pdecoeff.y30), kind="previous")

    # Mapping of inputs to feature configurations
    input_mapping = {
        3: lambda: np.vstack((g[:, 0], f2(g[:, 1]), g[:, 1])).T,
        4: lambda: np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), g[:, 1])).T,
        5: lambda: np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    }

    # Generate and return observations based on the number of inputs
    return input_mapping.get(nins, lambda: None)()


def load_weights(observers, conf, path):

    n = conf.parameters.nobs

    data = np.loadtxt(f"{path}_weights.txt")

    for j in range(n):
        observers[j]["weights"] = data[:, j+1].reshape(data[:, 0:1].shape)

    return observers

    
def scale_t(t):
    conf = compose(config_name='config_run')
    Troom = conf.temps.Troom
    Tmax = conf.temps.Tmax
    k = (t - Troom) / (Tmax - Troom)

    return round(k, n_digits)


def rescale_t(theta, conf=None):
    if conf is None:
        conf = compose(config_name='config_run')

    Troom = conf.temps.Troom
    Tmax = conf.temps.Tmax

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
    conf = compose(config_name='config_run')
    # Check if X is a single float value
    if isinstance(X, (int, float)):
        return X * conf.properties.L0

    # Iterate through each component in X and rescale if it is a list-like object
    rescaled_X = []
    for part in X:
        part = np.array(part, dtype=float)  # Ensure each part is converted into a numpy array
        rescaled_part = part * conf.properties.L0        # Apply the scaling
        rescaled_X.append(rescaled_part)    # Append rescaled part to the result list

    return rescaled_X


def rescale_time(tau):
    conf = compose(config_name='config_run')
    tau = np.array(tau)
    tf = conf.properties.tf
    j = tau*tf

    return np.round(j, 0)


def scale_time(t):
    conf = compose(config_name='config_run')
    tf = conf.properties.tf
    j = t/tf

    return np.round(j, n_digits)


def get_tc_positions():
    conf = compose(config_name='config_run')
    pars = conf.parameters
    Xy2 = 0.0
    Xy1 = 1.0

    # return {"y2": Xy2, "gt": pars.Xgt, "y1": Xy1}
    return {"y2": Xy2, "gt": pars.Xgt, "gt1": pars.Xgt1,"y1": Xy1}

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
    pars = config.parameters
    # props = config.model_properties
    pdecoeff = config.pdecoeff

    nobs = pars.nobs

    if nobs == 1:
        pdecoeff.wb = pars.wbobs
        setup_log(f"Training single observer with perfusion {pars.wbobs}.")
        output_model = train_model(config)
        return output_model

    multi_obs = []
    for j in range(nobs):
        obs = np.logspace(np.log10(pars.wbmin), np.log10(pars.wbmax), nobs)
        perf = obs[j]
        pdecoeff.wb = float(perf)
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
        index = np.abs(g[:, 0] - x_ref).argmin()
        rows_xref = g[g[:, 0] == g[:, 0][index]]   
        sys_xref = rows_xref[:, 2]

        if observers_data:
            nobs = len(observers_data)
            obs_err = np.abs(rows_xref[:, 3:3+nobs] - sys_xref[:, None])
            for i, observer in enumerate(observers_data):
                observer[f'obs_err_{round(x_ref, 2)}'] = obs_err[:, i]

        if mm_obs:
            mm_obs_err = np.abs(rows_xref[:, -1] - sys_xref)
            mm_obs[f'obs_err_{round(x_ref, 2)}'] = mm_obs_err
    
            # obs_dir = f"{tests_dir}/errs_meas"
            # np.savez(f"{obs_dir}/meas_cool_2_8_{round(x_ref, 2)}.npz", mm_obs_err)

    return observers_data, mm_obs


def get_pred(model, X, output_dir, label):

    y_sys_pinns = model.predict(X)
    data_to_save = np.column_stack((X[:, 0].round(n_digits), X[:, -1].round(n_digits), y_sys_pinns.round(n_digits)))
    np.savetxt(f'{output_dir}/prediction_{label}.txt', data_to_save, fmt='%.6f %.6f %.6f', delimiter=' ') 

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
    pars = conf.parameters
    nobs = pars.nobs
    preds = [x_obs[:, 0], x_obs[:, -1]]

    # Process for multiple observers
    if isinstance(multi_obs, list):
        for el in range(nobs):
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
            "label": f"observer_{i if nobs > 1 else pars.obsindex}"
        }
        for i in range(nobs) 
    ]

    mm_obs = None

    obs_dict, mm_obs = compute_obs_err(ground_truth, obs_dict, mm_obs)

    if nobs==1:
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


def load_observers_preds(ground_truth, conf, label):
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
    pars = conf.parameters
    nobs = pars.nobs
    output_dir = conf.experiment.pred_fold


    mm_obs=None
    obs_dict = []

    if conf.plot.show_obs ==True:
        for i in range(nobs):
            file_path = os.path.join(output_dir, f"observer_{i}_{label}.txt")
            data = np.loadtxt(file_path)
            obs_dict.append({
                "grid": data[:, :2],
                "theta": data[:, 2],
                "label": f"observer_{i}"
            })
        
        
        obs_dict, _ = compute_obs_err(ground_truth, obs_dict, mm_obs)
        weights_path = f"{output_dir}/{label}"
        obs_dict = load_weights(obs_dict, conf, weights_path)

    mm_obs_file_path = os.path.join(output_dir, f"multi_observer_{label}.txt")
    mm_obs_data = np.loadtxt(mm_obs_file_path)
    mm_obs = {
        "grid": mm_obs_data[:, :2],
        "theta": mm_obs_data[:, 2],
        "label": "multi_observer"
    }

    _, mm_obs = compute_obs_err(ground_truth, obs_dict, mm_obs)
    obs_dict, mm_obs = calculate_l2(ground_truth, obs_dict, mm_obs)

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


def get_plot_params(conf_run):
    """
    Load plot parameters based on configuration for each entity (system, observers, etc.),
    and set up characteristics such as colors, linestyles, linewidths, and alphas.
    
    :param conf: Configuration object loaded from YAML.
    :return: Dictionary containing plot parameters for each entity.
    """
    conf = OmegaConf.load(f"{src_dir}/configs/plot.yaml")

    entities = conf.entities

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
    nobs = conf_run.parameters.nobs
    observer_params = {}
    observer_gt_params = {}
    meas_points_params = getattr(entities, "meas_points")
    losses_params = getattr(entities, "losses")

    if 1<=nobs<=8:
        for j in range(nobs):
            i = conf_run.parameters.obsindex if nobs == 1 else j
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
    pars = conf.parameters
    nobs = pars.nobs
    lam = pars.lam
    ups = pars.upsilon

    p0 = np.full((nobs,), 1/nobs)

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
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd(f"{src_dir}/matlab", nargout=0)
    eng.BioHeat(nargout=0)
    eng.quit()
    setup_log("MATLAB ground truth calculation completed.")
    print("MATLAB ground truth calculation completed.")


def compute_y_theory(grid, sys, obs, conf):
    pars = conf.parameters
    str = np.where(np.abs(pars.wbsys - pars.wbobs) <= 1e-08, 'exact', 'diff')
    x = np.unique(grid[:, 0])
    t = np.unique(grid[:, -1])
    sys_0 = sys[:len(x)]
    sys_0 = sys_0.reshape(len(sys_0), 1)
    obs_0 = obs[:len(x)]
    obs_0 = obs_0.reshape(len(obs_0), 1)
    l2_0 = calculate_l2(grid[grid[:, 1]==0], sys_0, obs_0)

    decay = getattr(pars, f"dr{str}")

    theory = {"grid": grid, "theta": l2_0 * np.exp(-t*decay), "label": "theory"}
    bound = {"grid": grid, "theta": np.full_like(t, pars.c0), "label": "bound"}
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
            a = dicts[i]['theta'].reshape(theta_obsvs[:, i-1].shape)
            theta_obsvs[:, i-1] = a
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
    # theta_obsvs=np.array(theta_obsvs, dtype=float).reshape(len(result), len(dicts))
    result = np.hstack((result, theta_obsvs))

    return result



