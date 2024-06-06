import deepxde as dde
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import wandb
import glob
# import rff
import json
from tqdm import tqdm
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from kan import KAN, LBFGS

# device = torch.device("cpu")
device = torch.device("cuda")

name = None
run_dir = None
model_dir = None
figures_dir = None
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)


def set_name(prj, run):
    global name, run_dir, model_dir, figures_dir
    name = prj
    name_dir = os.path.join(tests_dir, name)
    os.makedirs(name_dir, exist_ok=True)

    run_dir = os.path.join(name_dir, run)
    os.makedirs(run_dir, exist_ok=True)

    model_dir = os.path.join(run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    return name, run_dir, model_dir, figures_dir


def read_config(run):
    filename = f"{model_dir}/config_{run}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            config = json.load(file)
    else:
        # Create default config if file doesn't exist
        config = create_default_config()
        write_config(config, run)
    return config


def create_default_config():
    # Define default configuration parameters
    default_config = {
        "activation": "tanh",
        "convection_coefficient": 350,
        "d": 0.03,
        "initial_weights_regularizer": True,
        "initialization": "Glorot normal",
        "iterations": 30000,
        "LBFGS": False,
        "learning_rate": 0.001,
        "num_dense_layers": 2,
        "num_dense_nodes": 50,
        "output_injection_gain": 5,
        "power": 0,
        "perfusion": False,
        "resampling": True,
        "resampler_period": 100,
        "rhoc": 4181000,
        "thermal_cond": 0.563
    }
    return default_config


def write_config(config, run):
    filename = f"{model_dir}/config_{run}.json"
    with open(filename, 'w') as file:
        json.dump(config, file, indent=4)


def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(0)
    return losshistory.loss_train[0]


def restore_model(name):
    conf=read_config(name)
    model = create_default_config(name)
    LBFGS = conf["LBFGS"]

    matching_files = glob.glob(f"{model_dir}/{name}-*.pt")
    if matching_files:
        # If there are multiple matching files, sort them to ensure consistency
        matching_files.sort()
        # Select the first matching file
        selected_file = matching_files[0]
    else:
        print("No matching files found.")   

    if LBFGS:
        model.compile("L-BFGS")
        model.restore(selected_file, verbose=0)
    else:
        model.restore(selected_file, verbose=0)

    return model


def plot_loss_components(losshistory):
    loss_train = losshistory.loss_train
    loss_test = losshistory.loss_test
    matrix = np.array(loss_train)
    test = np.array(loss_test).sum(axis=1).ravel()
    train = np.array(loss_train).sum(axis=1).ravel()
    loss_res = matrix[:, 0]
    loss_bc0 = matrix[:, 1]
    loss_bc1 = matrix[:, 2]    
    loss_ic = matrix[:, 3]

    fig = plt.figure(figsize=(6, 5))
    # iters = np.arange(len(loss_ic))
    iters = losshistory.steps
    with sns.axes_style("darkgrid"):
        plt.clf()
        plt.plot(iters, loss_res, label=r'$\mathcal{L}_{res}$')
        plt.plot(iters, loss_bc0, label=r'$\mathcal{L}_{bc0}$')
        plt.plot(iters, loss_bc1, label=r'$\mathcal{L}_{bc1}$')
        plt.plot(iters, loss_ic, label=r'$\mathcal{L}_{ic}$')
        plt.plot(iters, test, label='test loss')
        plt.plot(iters, train, label='train loss')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/losses.png")
        plt.close()
    

def compute_metrics(true, pred):

    if np.isnan(pred.any()):
        metrics = {
            "MSE": "ErrorNan",
            "MAPE": "ErrorNan",
            "L2RE": "ErrorNan",
            "max_APE": "ErrorNan"}
        
    elif not np.isfinite(pred.all()):
        metrics = {
            "MSE": "ErrorInf",
            "MAPE": "ErrorInf",
            "L2RE": "ErrorInf",
            "max_APE": "ErrorInf"}
        
    else:

        MSE = dde.metrics.mean_squared_error(true, pred)
        MAPE = np.mean(np.abs((true - pred)/true))*100
        L2RE = dde.metrics.l2_relative_error(true, pred)        # For relative accuracy
        max_APE = np.max(np.abs((true - pred)/true))*100        # For the worst case scenario

        metrics = {
            "MSE": MSE,
            "MAPE": MAPE,
            "L2RE": L2RE,
            "max_APE": max_APE,
        }

    return metrics


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def bc0_obs(x, theta, X):
    return x[:, 1:2] - theta


def create_nbho(config):
    k_th, rhoc, d, h, K = config["thermal_cond"], config["rhoc"],config["d"], config["convection_coefficient"], config["output_injection_gain"] 
    s, W = config["power"], config["perfusion"]
    activation = config["activation"]
    initial_weights_regularizer = config["initial_weights_regularizer"]
    initialization = config["initialization"]
    learning_rate = config["learning_rate"]
    num_dense_layers = config["num_dense_layers"]
    num_dense_nodes = config["num_dense_nodes"]

    L_0, tauf = 0.15, 3600
    T_max, T_min = 35, 22
    dT = T_max - T_min
    cb = 3825

    D = d/L_0
    alpha = k_th/rhoc

    C1, C2 = tauf/L_0**2, dT*tauf/rhoc
    C3 = C2*dT*cb

    if W:
        def pde(x, y):
            dy_t = dde.grad.jacobian(y, x, i=0, j=4)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            # Backend tensorflow.compat.v1 or tensorflow
            return (
                dy_t
                - alpha * C1 * dy_xx - C2 * s*torch.exp(-x[:, 0:1]/D) + C3 * W *y
            )
    
    else:
        def pde(x, y):
            dy_t = dde.grad.jacobian(y, x, i=0, j=4)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            # Backend tensorflow.compat.v1 or tensorflow
            return (
                dy_t
                - alpha * C1 * dy_xx - C2 * s*torch.exp(-x[:, 0:1]/D)
            )

    def bc1_obs(x, theta, X):
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
        return dtheta_x - h*(x[:, 3:4]-x[:, 2:3]) - K * (x[:, 2:3] - theta)


    def ic_obs(x):

        z = x[:, 0:1]
        y1 = x[:, 1:2]
        y2 = x[:, 2:3]
        y3 = x[:, 3:4]
        beta = h * (y3 - y2) + K * (y2 -y1)
        a2 = 5.0

        e = y1 + ((beta - ((2/L_0)+K)*a2)/((1/L_0)+K))*z + a2*z**2
        return e

    xmin = [0, 0, 0, 0]
    xmax = [1, 1, 1, 1]
    geom = dde.geometry.Hypercube(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.OperatorBC(geomtime, bc0_obs, boundary_0)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)

    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_0, bc_1, ic],
        num_domain=2560,
        num_boundary=200,
        num_initial=100,
        num_test=10000,
    )

    layer_size = [5] + [num_dense_nodes] * num_dense_layers + [1]
    net = dde.nn.FNN(layer_size, activation, initialization)

    model = dde.Model(data, net)

    if initial_weights_regularizer:
        initial_losses = get_initial_loss(model)
        loss_weights = len(initial_losses) / initial_losses
        model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    else:
        model.compile("adam", lr=learning_rate)
    return model


def train_model(name):
    conf = read_config(name)
    mm = create_nbho(name)

    LBFGS = conf["LBFGS"]
    epochs = conf["iterations"]
    ini_w = conf["initial_weights_regularizer"]
    resampler = conf["resampling"]
    resampler_period = conf["resampler_period"]

    if LBFGS:
        optim, iters = "lbfgs", "*"
        # mm.compile("L-BFGS")
    else:
        optim, iters = "adam", epochs

    # If it already exists a model with the exact config, restore it
    trained_models = glob.glob(f"{model_dir}/{optim}-{iters}.pt")
    if trained_models:
        trained_models.sort()
        trained_model = trained_models[0]
        if LBFGS:
            mm.compile("L-BFGS")
        mm.restore(trained_model, verbose=0)
    else:
        if resampler:
            pde_residual_resampler = dde.callbacks.PDEPointResampler(period=resampler_period)
            callbacks = [pde_residual_resampler]
        else:
            callbacks = []


        if LBFGS:
            matching_files = glob.glob(f"{model_dir}/adam-{epochs}.pt")
            if matching_files:
                matching_files.sort()
                selected_file = matching_files[0]
                mm.restore(selected_file, verbose=0)
            else:
                mm.train(
                callbacks=callbacks, 
                iterations = epochs,
                model_save_path=f"{model_dir}/adam", 
                display_every=conf.get("display_every", 1000)
                )
            if ini_w:
                initial_losses = get_initial_loss(mm)
                loss_weights = 5 / initial_losses
                mm.compile("L-BFGS", loss_weights=loss_weights)
            else:
                mm.compile("L-BFGS")

            losshistory, train_state = mm.train(
                callbacks=callbacks, 
                model_save_path=f"{model_dir}/lbfgs", 
                display_every=conf.get("display_every", 1000)
            )
            plot_loss_components(losshistory)
        else:
            losshistory, train_state = mm.train(
                iterations=epochs, 
                callbacks=callbacks, 
                model_save_path=f"{model_dir}/adam", 
                display_every=conf.get("display_every", 1000)
            )
            plot_loss_components(losshistory)

    # Compute metrics
    # metrics = plot_1d(mm, name)

    return mm#, metrics
