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
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D

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


properties = {
    "L0": None, "tauf": None, "k": None, "p0": None, "d": None,
    "rhoc": None, "cb": None, "h": None, "Tmin": None, "Tmax": None, "alpha": None,
    "W": None, "steep": None, "tchange": None
}



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


def get_properties(n):
    global L0, tauf, k, p0, d, rhoc, cb, h, Tmin, Tmax, alpha, W, steep, tchange
    file_path = os.path.join(src_dir, 'simulations', f'data{n}.json')

    # Open the file and load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    properties.update(data['Parameters'])
    par = data['Parameters']
    local_vars = locals()
    for key in par:
        if key in local_vars:
            local_vars[key] = par[key]

    L0, tauf, k, p0, d, rhoc, cb, h, Tmin, Tmax, alpha, W, steep, tchange = (
        par["L0"], par["tauf"], par["k"], par["p0"], par["d"], par["rhoc"],
        par["cb"], par["h"], par["Tmin"], par["Tmax"], par["alpha"], par["W"], par["steep"], par["tchange"]
    )


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
    network = {
        "activation": "tanh", 
        "initial_weights_regularizer": True, 
        "initialization": "Glorot normal",
        "iterations": 30000,
        "LBFGS": False,
        "learning_rate": 0.001,
        "num_dense_layers": 2,
        "num_dense_nodes": 50,
        "output_injection_gain": 50,
        "resampling": True,
        "resampler_period": 100
    }
    return network

def write_config(config, run):
    filename = f"{model_dir}/config_{run}.json"
    with open(filename, 'w') as file:
        json.dump(config, file, indent=4)


def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(0)
    return losshistory.loss_train[0]


# def restore_model(name):
#     conf=read_config(name)
#     model = create_default_config(name)
#     LBFGS = conf["LBFGS"]

#     matching_files = glob.glob(f"{model_dir}/{name}-*.pt")
#     if matching_files:
#         # If there are multiple matching files, sort them to ensure consistency
#         matching_files.sort()
#         # Select the first matching file
#         selected_file = matching_files[0]
#     else:
#         print("No matching files found.")   

#     if LBFGS:
#         model.compile("L-BFGS")
#         model.restore(selected_file, verbose=0)
#     else:
#         model.restore(selected_file, verbose=0)

#     return model


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


def create_nbho(name):
    net = read_config(name)

    activation = net["activation"]
    initial_weights_regularizer = net["initial_weights_regularizer"]
    initialization = net["initialization"]
    learning_rate = net["learning_rate"]
    num_dense_layers = net["num_dense_layers"]
    num_dense_nodes = net["num_dense_nodes"]
    K = net["output_injection_gain"]

    dT = Tmax - Tmin

    D = d/L0
    alpha = k/rhoc

    C1, C2 = tauf/L0**2, dT*tauf/rhoc
    C3 = C2*dT*cb

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=4)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        # Backend tensorflow.compat.v1 or tensorflow
        return (
            dy_t
            - alpha * C1 * dy_xx - C2 * p0*torch.exp(-x[:, 0:1]/D) + C3 * W *y
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
        a2 = -0.7

        e = y1 + ((beta - ((2/L0)+K)*a2)/((1/L0)+K))*z + a2*z**2
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

    return mm


def gen_testdata(n):
    data = np.loadtxt(f"{src_dir}/simulations/file{n}.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    return X, y

def gen_obsdata(n):
    g = np.hstack((gen_testdata(n)))
    instants = np.unique(g[:, 1])

    rows_0 = g[g[:, 0] == 0.0]
    rows_1 = g[g[:, 0] == 1.0]

    y1 = rows_0[:, -1].reshape(len(instants),)
    f1 = interp1d(instants, y1, kind='previous')

    y2 = rows_1[:, -1].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    def f3(ii):
        return ii + (1 - ii)/(1 + np.exp(-20*(ii - 0.25)))
    
    # tm = 0.9957446808510638
    # if tau > tm:
    #     tau = tm

    Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    return Xobs


def plot(model, n_test):
    e, theta_true = gen_testdata(n_test)
    g = gen_obsdata(n_test)

    theta_pred = model.predict(g)
    plot_comparison(e, theta_true, theta_pred)
    plot_l2(e, theta_true, theta_pred)
    # plot_tf(e, theta_true, theta_pred)


def plot_comparison(e, theta_true, theta_pred):

    la = len(np.unique(e[:, 0]))
    le = len(np.unique(e[:, 1]))

    # Predictions
    fig = plt.figure(3, figsize=(9, 4))

    col_titles = ['Measured', 'Observed', 'Error']
    surfaces = [
        [theta_true.reshape(le, la), theta_pred.reshape(le, la),
            np.abs(theta_true - theta_pred).reshape(le, la)]
    ]

    # Create a grid of subplots
    grid = plt.GridSpec(1, 3)

    # Iterate over columns to add subplots
    for col in range(3):
        ax = fig.add_subplot(grid[0, col], projection='3d')
        configure_subplot(ax, e, surfaces[0][col])

        # Set column titles
        ax.set_title(col_titles[col], fontsize=8, y=.96, weight='semibold')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.15)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/comparison.png")
    plt.show()
    plt.clf()

def plot_l2(e, theta_true, theta_pred):
    t = np.unique(e[:, 1])
    l2 = []
    tot = np.hstack(e, theta_true, theta_pred)

    for el in t:
        df = tot[tot[:, 1]==el]
        l2.append(dde.metrics.l2_relative_error(df[:, 2], df[:, 3]))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(t, l2, alpha=1.0, linewidth=1.8, color='C0')

    ax1.set_xlabel(xlabel=r"Time t", fontsize=7)  # xlabel
    ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)  # ylabel
    ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlim(0, 1.01)
    plt.yticks(fontsize=7)

    plt.grid()
    ax1.set_box_aspect(1)
    plt.savefig(f"{figures_dir}/l2.png")
    plt.show()
    plt.clf()

# def plot_tf(e, theta_true, theta_pred):

#     plt.tight_layout()
#     plt.savefig(f"{figures_dir}/tf.png")
#     plt.show()
#     plt.clf()

def configure_subplot(ax, XS, surface):
    la = len(np.unique(XS[:, 0:1]))
    le = len(np.unique(XS[:, 1:]))
    X = XS[:, 0].reshape(le, la)
    T = XS[:, 1].reshape(le, la)

    ax.plot_surface(X, T, surface, cmap='inferno', alpha=.8)
    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.dist = 10
    ax.view_init(20, -120)

    # Set axis labels
    ax.set_xlabel('Depth', fontsize=7, labelpad=-1)
    ax.set_ylabel('Time', fontsize=7, labelpad=-1)
    ax.set_zlabel('Theta', fontsize=7, labelpad=-4)