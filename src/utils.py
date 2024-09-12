import deepxde as dde
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import wandb
import glob
import json
from scipy.interpolate import interp1d
from scipy import integrate
import pickle
import pandas as pd
import coeff_calc as cc
import hashlib
import joblib 

dde.config.set_random_seed(200)

# device = torch.device("cpu")
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)


models = os.path.join(git_dir, "models")
os.makedirs(models, exist_ok=True)

figures = os.path.join(tests_dir, "figures")
os.makedirs(figures, exist_ok=True)


prj_figs, run_figs = [None]*2


f1, f2, f3 = [None]*3

def set_prj(prj):
    global prj_figs

    prj_figs = os.path.join(figures, prj)
    os.makedirs(prj_figs, exist_ok=True)

    return prj_figs


def set_run(run):
    global prj_figs, run_figs

    run_figs = os.path.join(prj_figs, run)
    os.makedirs(run_figs, exist_ok=True)

    return run_figs


def read_json(filename):
    filepath = f"{src_dir}/{filename}"
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
    return data


def write_json(data, filename):
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_data = {k: convert_to_serializable(v) for k, v in data.items()}

    filepath = f"{src_dir}/{filename}"
    with open(filepath, 'w') as file:
        json.dump(serializable_data, file, indent=4)


def generate_config_hash(config_data):
    config_string = json.dumps(config_data, sort_keys=True)  # Sort to ensure consistent ordering
    config_hash = hashlib.md5(config_string.encode()).hexdigest()  # Create a unique hash
    return config_hash


def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(0)
    return losshistory.loss_train[0]


def plot_loss_components(losshistory):
    global models
    prop = read_json("properties.json")
    hash = generate_config_hash(prop)
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
        plt.savefig(f"{models}/losses_{hash}.png", dpi=120)
        plt.close()
    

def compute_metrics(true, pred):
    small_number = 1e-3
    
    true = np.ravel(true)
    pred = np.ravel(pred)
    true_nonzero = np.where(true != 0, true, small_number)
    
    L2RE = dde.metrics.l2_relative_error(true, pred)
    MSE = dde.metrics.mean_squared_error(true, pred)
    max_err = np.max(np.abs((true_nonzero - pred)))
    mean_err = np.mean(np.abs((true_nonzero - pred)))
    
    metrics = {
        "L2RE": L2RE,
        "MSE": MSE,
        "max": max_err,
        "mean": mean_err,
    }
    return metrics


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def output_transform(x, y):
    return x[:, 0:1] * y

def create_nbho():
    net = read_json("config.json")
    properties = read_json("properties.json")

    activation = net["activation"]
    initial_weights_regularizer = net["initial_weights_regularizer"]
    initialization = net["initialization"]
    learning_rate = net["learning_rate"]
    num_dense_layers = net["num_dense_layers"]
    num_dense_nodes = net["num_dense_nodes"]

    a1 = cc.a1
    a2 = cc.a2
    a3 = cc.a3
    K = properties["K"]
    delta = properties["delta"]
    W = properties["W"]


    def pde(x, theta):
        dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=4)
        dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)

        return (
            a1 * dtheta_tau
            - dtheta_xx + W * a2 * theta
        )
    
    def ic_obs(x):
        z = x[:, 0:1]
        y1_0 = x[:, 1:2]
        y2_0 = x[:, 2:3]
        y3_0 = x[:, 3:4]

        b1 = (a3*y3_0+(K-a3)*y2_0-(2+K)*delta)/(1+K)

        return y1_0 + b1*z + delta*z**2

    def bc1_obs(x, theta, X):

        return theta - x[:, 1:2]

    def bc0_obs(x, theta, X):
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)

        return dtheta_x - a3 * (x[:, 3:4] - x[:, 2:3]) - K * (x[:, 2:3] - theta)

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
        lambda x, theta: pde(x, theta),
        [bc_0, bc_1, ic],
        num_domain=2560,
        num_boundary=200,
        num_initial=100,
        num_test=10000,
    )

    layer_size = [5] + [num_dense_nodes] * num_dense_layers + [1]
    net = dde.nn.FNN(layer_size, activation, initialization)

    # net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    if initial_weights_regularizer:
        initial_losses = get_initial_loss(model)
        loss_weights = len(initial_losses) / initial_losses
        model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    else:
        model.compile("adam", lr=learning_rate)

    return model


def train_model():
    global models
    conf = read_json("properties.json")
    config_hash = generate_config_hash(conf)
    model_path = os.path.join(models, f"model_{config_hash}.pt")

    mm = create_nbho()

    if os.path.exists(model_path):
        # Model exists, load it
        print(f"Loading model from {model_path}")
        mm.restore(model_path, verbose=0)
    
    LBFGS = conf["LBFGS"]
    ini_w = conf["initial_weights_regularizer"]
    resampler = conf["resampling"]
    resampler_period = conf["resampler_period"]

    callbacks = [dde.callbacks.PDEPointResampler(period=resampler_period)] if resampler else []

    if LBFGS:
        if ini_w:
            initial_losses = get_initial_loss(mm)
            loss_weights = len(initial_losses) / initial_losses
            mm.compile("L-BFGS", loss_weights=loss_weights)
        else:
            mm.compile("L-BFGS")
        
        losshistory, train_state = train_and_save_model(mm, callbacks)
    else:
        losshistory, train_state = train_and_save_model(mm, callbacks)

    plot_loss_components(losshistory)
    return mm


def train_and_save_model(model, callbacks):
    global models

    conf = read_json("properties.json")
    config_hash = generate_config_hash(conf)
    model_path = os.path.join(models, f"model_{config_hash}.pt")

    conf = read_json("properties.json")
    display_every = 1000
    losshistory, train_state = model.train(
        iterations=conf["iterations"],
        callbacks=callbacks,
        model_save_path=model_path,
        display_every=display_every
    )

    config_path = model_path.replace(".pt", ".json")
    with open(config_path, 'w') as f:
        json.dump(conf, f, indent=4)

    return losshistory, train_state


def gen_testdata():
    data = np.loadtxt(f"{src_dir}/output_matlab.txt")
    x, t, sys, obs, mmobs = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T, data[:, 3:11].T, data[:, 11:12].T
    X = np.vstack((x, t)).T
    y_sys = sys.flatten()[:, None]
    y_mmobs = mmobs.flatten()[:, None]

    return X, y_sys, obs.T, y_mmobs

def load_weights():
    data = np.loadtxt(f"{src_dir}/weights_matlab.txt")
    t, weights = data[:, 0:1], data[:, 1:9].T
    return t, np.array(weights)


def gen_obsdata():
    global f1, f2, f3
    g = np.hstack((gen_testdata()))
    instants = np.unique(g[:, 1])
    
    rows_1 = g[g[:, 0] == 1.0]
    rows_0 = g[g[:, 0] == 0.0]

    y1 = rows_0[:, 2].reshape(len(instants),)
    f1 = interp1d(instants, y1, kind='previous')

    y2 = rows_1[:, 2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    properties = read_json("properties.json")
    theta_w = scale_t(properties["Twater"])
    y3 = np.full_like(y2, theta_w)
    f3 = interp1d(instants, y3, kind='previous')

    Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    return Xobs


def load_from_pickle(file_path):
    with open(file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)
    
def scale_t(t):

    prop = read_json("properties.json")
    Troom = prop["Troom"]
    Tmax = prop["Tmax"]

    return (t - Troom) / (Tmax - Troom)
    
def import_testdata():
    df = load_from_pickle(f"{src_dir}/cooling_scaled.pkl")

    x_tcs = np.linspace(0, 1, num=8).round(4)
    x_y1 = x_tcs[7]
    x_gt1 = x_tcs[4]
    x_gt2 = x_tcs[1]
    x_y2 = x_tcs[0]

    positions = [x_y1, x_gt1, x_gt2, x_y2]

    dfs = []
    boluses = []
    for time_value in df['tau']:

        # Extract 'theta' values for the current 'time' from df_result
        theta_values = df[df['tau'] == time_value][
            ['y1', 'gt1', 'gt2', 'y2']].values.flatten()

        time_array = np.array([positions, [time_value] * 4, theta_values]).T
        bol_value = df[df['tau'] == time_value][['y3']].values.flatten()

        bolus_array = np.array([bol_value]*4)
        
        boluses.append(bolus_array)
        dfs.append(time_array)

    vstack_array = np.vstack(dfs)
    boluses_arr = np.vstack(boluses)

    return np.hstack((vstack_array,boluses_arr))


def import_obsdata():
    global f1, f2, f3
    g = import_testdata()
    instants = np.unique(g[:, 1])

    rows_1 = g[g[:, 0] == 1.0]
    rows_0 = g[g[:, 0] == 0.0]

    y1 = rows_0[:, -2].reshape(len(instants),)
    f1 = interp1d(instants, y1, kind='previous')

    y2 = rows_1[:, -2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    y3 = rows_1[:, -1].reshape(len(instants),)
    f3 = interp1d(instants, y3, kind='previous')

    Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    return Xobs


def plot_and_metrics(model):

    o = import_testdata()
    e, theta_true = o[:, 0:2], o[:, -2]
    g = import_obsdata()

    theta_pred = model.predict(g)

    plot_comparison(e, theta_true, theta_pred)
    # check_obs(e, theta_obs, theta_pred)
    plot_l2_tf(e, theta_true, theta_pred, model)
    # plot_tf(e, theta_true, model)
    metrics = compute_metrics(theta_true, theta_pred)
    return metrics


def check_obs(e, theta_true, theta_pred):
    global run_figs, prj_figs

    la = len(np.unique(e[:, 0]))
    le = len(np.unique(e[:, 1]))

    # Predictions
    fig = plt.figure(3, figsize=(9, 4))

    col_titles = ['MATLAB', 'PINNs', 'Error']
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

    plt.savefig(f"{run_figs}/check_obs.png", dpi=120)

    # plt.show()
    plt.close()
    # plt.clf()


def plot_comparison(e, t_true, t_pred, MObs=False):
    global run_figs, prj_figs

    la = len(np.unique(e[:, 0]))
    le = len(np.unique(e[:, 1]))

    theta_true = t_true.reshape(le, la)
    theta_pred = t_pred.reshape(le, la)

    # Predictions
    fig = plt.figure(3, figsize=(9, 4))

    col_titles = ['Measured', 'Observed', 'Error']
    surfaces = [
        [theta_true, theta_pred,
            np.abs(theta_true - theta_pred)]
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

    # plt.tight_layout()

    if MObs:
        plt.savefig(f"{prj_figs}/comparison.png", dpi=120)

    else:
        plt.savefig(f"{run_figs}/comparison.png", dpi=120)

    # plt.show()
    plt.close()
    # plt.clf()


def plot_l2_norm(e, theta_true, theta_pred):
    t = np.unique(e[:, 1])
    l2 = []
    t_filtered = t[t > 0.0001]

    theta_true = theta_true.reshape(len(e), 1)
    theta_pred = theta_pred.reshape(len(e), 1)
    tot = np.hstack((e, theta_true, theta_pred))
    t = t_filtered

    for el in t:
        df = tot[tot[:, 1] == el]
        l2.append(dde.metrics.l2_relative_error(df[:, 2], df[:, 3]))

    fig = plt.figure(figsize=(10, 5))  # Adjust the size as needed
    ax1 = fig.add_subplot(121)
    ax1.plot(t, l2, alpha=1.0, linewidth=1.2, color='C0')
    ax1.grid()

    ax1.set_xlabel(xlabel=r"Time t", fontsize=7)  # xlabel
    ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)  # ylabel
    ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlim(0, 1.01)
    ax1.set_box_aspect(1)

    return fig, ax1


def plot_l2_tf(e, theta_true, theta_pred, model):
    global run_figs
    e, theta_true, theta_pred = e.reshape((len(e), 2)), theta_true.reshape((len(e), 1)), theta_pred.reshape((len(e), 1))

    fig, ax1 = plot_l2_norm(e, theta_true, theta_pred)

    tot = np.hstack((e, theta_true))
    final = tot[tot[:, 1]==1]
    xtr = np.unique(tot[:, 0])
    x = np.linspace(0, 1, 100)
    true = final[:, -1]

    Xobs = np.vstack((x, f1(np.ones_like(x)), f2(np.ones_like(x)), f3(np.ones_like(x)), np.ones_like(x))).T
    pred = model.predict(Xobs)

    ax2 = fig.add_subplot(122)
    ax2.plot(xtr, true, marker="x", linestyle="None", alpha=1.0, color='C0', label="true")
    ax2.plot(x, pred, alpha=1.0, linewidth=1.0, color='C2', label="pred")

    ax2.set_xlabel(xlabel=r"Space x", fontsize=7)  # xlabel
    ax2.set_ylabel(ylabel=r"$\Theta$", fontsize=7)  # ylabel
    ax2.set_title(r"Prediction at tf", fontsize=7, weight='semibold')
    ax2.set_ylim(bottom=0.0)
    ax2.set_xlim(0, 1.01)
    ax2.legend()
    plt.yticks(fontsize=7)

    plt.grid()
    ax2.set_box_aspect(1)
    plt.savefig(f"{run_figs}/l2_tf.png", dpi=120)
    
    # plt.show()
    plt.close()
    # plt.clf()


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



def mm_observer():

    data =  read_json("parameters.json")

    W0, W1, W2, W3, W4, W5, W6, W7 = data["W0"], data["W1"], data["W2"], data["W3"], data["W4"], data["W5"], data["W6"], data["W7"]

    obs = np.array([W0, W1, W2, W3, W4, W5, W6, W7])
    # a2_obs = np.dot(cc.a2, obs).round(4)

    n_obs = len(obs)

    # wandb.init(
    #     project=name_prj, name=name_run,
    #     config=read_config(name_run)
    # )

    multi_obs = []
    
    for j in range(n_obs):
        # run = f"n{j}_W{a2_obs[j]}"
        # set_run(run)
        # config = read_config()
        perf = obs[j]
        properties = read_json("properties.json")
        properties["W"] = perf
        write_json(properties, "properties.json")

        model = train_model()
        multi_obs.append(model)


    return multi_obs


def mu(o, tau):
    global f1, f2, f3
    net = read_json("parameters.json")
    upsilon = net["upsilon"]

    xo = np.vstack((np.ones_like(tau), f1(tau), f2(tau), f3(tau), tau)).T
    muu = []
    for el in o:
        oss = el.predict(xo)
        scrt = upsilon*np.abs(oss-f2(tau))**2
        muu.append(scrt)
    muu = np.array(muu).reshape(len(muu),)
    return muu


def compute_mu():
    net = read_json("parameters.json")
    upsilon = net["upsilon"]
    g = np.hstack((gen_testdata()))

    rows_0 = g[g[:, 0] == 0.0]
    sys_0 = rows_0[:, 2:3]
    obss_0 = rows_0[:, 3:11]
    sys_0 = sys_0.reshape(obss_0[:, 0].shape)
    e = np.abs(obss_0[:, 0] - sys_0)

    muu = []
    # Loop through each column of obss_0
    for el in range(obss_0.shape[1]):
        
        # Compute mu for each element and append to the list
        mu_value = upsilon * np.abs(obss_0[:, el] - sys_0)**2
        muu.append(mu_value)

    return np.array(muu)


def plot_weights(weights, t, gt=False):
    global run_figs
    param = read_json("parameters.json")
    lam = param["lambda"]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    colors = ['C3', 'lime', 'blue', 'aqua', 'm', 'darkred', 'k', 'yellow']

    for i in range(weights.shape[0]):
        # plt.plot(tauf * t, x[i], alpha=1.0, linewidth=1.8, color=colors[i], label=f"Weight $p_{i+1}$")
        plt.plot(t, weights[i], alpha=1.0, linewidth=1.0, color=colors[i], label=f"Weight $p_{i}$")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=0.0)

    ax1.set_xlabel(xlabel=r"Time t")  # xlabel
    ax1.set_ylabel(ylabel=r"Weights $p_j$")  # ylabel
    ax1.legend()
    ax1.set_title(r"Dynamic weights, $\lambda=$"f"{lam}", weight='semibold')
    plt.grid()
    if gt:
        plt.savefig(f"{run_figs}/weights_lam_{lam}_matlab.png", dpi=120, bbox_inches='tight')
    else:
        plt.savefig(f"{run_figs}/weights_lam_{lam}.png", dpi=120, bbox_inches='tight')

    # plt.show()
    plt.close()
    # plt.clf()


def plot_mu(multi_obs, t):
    global prj_figs
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    muy = []
    for el in t:
        muy.append(mu(multi_obs, el))

    mus = np.array(muy)
    colors = ['C3', 'lime', 'blue', 'aqua', 'm', 'darkred', 'k', 'yellow']
    for i in range(mus.shape[1]):
        # plt.plot(tauf * t, x[i], alpha=1.0, linewidth=1.8, color=colors[i], label=f"Weight $p_{i+1}$")
        plt.plot(t, mus[:, i], alpha=1.0, linewidth=1.0, color=colors[i], label=f"$e_{i}$")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=0.0)

    ax1.set_xlabel(xlabel=r"Time t")  # xlabel
    ax1.set_ylabel(ylabel=r"Error")  # ylabel
    ax1.legend()
    ax1.set_title(r"Observation errors", weight='semibold')
    plt.grid()
    plt.savefig(f"{prj_figs}/obs_error.png", dpi=120, bbox_inches='tight')

    # plt.show()
    plt.close()
    # plt.clf()

def plot_mu_gt(mus, t):
    global run_figs
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    colors = ['C3', 'lime', 'blue', 'aqua', 'm', 'darkred', 'k', 'yellow']
    for i in range(mus.shape[0]):
        # plt.plot(tauf * t, x[i], alpha=1.0, linewidth=1.8, color=colors[i], label=f"Weight $p_{i+1}$")
        plt.plot(t, mus[i], alpha=1.0, linewidth=1.0, color=colors[i], label=f"$e_{i}$")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=0.0)

    ax1.set_xlabel(xlabel=r"Time t")  # xlabel
    ax1.set_ylabel(ylabel=r"Error")  # ylabel
    ax1.legend()
    ax1.set_title(r"Observation errors", weight='semibold')
    plt.grid()
    plt.savefig(f"{run_figs}/obs_error_matlab.png", dpi=120, bbox_inches='tight')

    # plt.show()
    plt.close()
    # plt.clf()


def mm_plot_and_metrics(multi_obs, lam, n):
    e, theta_true = gen_testdata(n)
    g = gen_obsdata(n)
    # a = import_testdata()
    # e = a[:, 0:2]
    # theta_true = a[:, 2]
    # g = import_obsdata()

    theta_pred = mm_predict(multi_obs, lam, g).reshape(theta_true.shape)

    plot_comparison(e, theta_true, theta_pred, MObs=True)
    mm_plot_l2_tf(e, theta_true, theta_pred, multi_obs, lam)

    metrics = compute_metrics(theta_true, theta_pred)
    return metrics


def mm_predict(multi_obs, lam, g):
    global prj_logs
    a = np.load(f'{prj_logs}/weights_lam_{lam}.npy', allow_pickle=True)
    weights = a[1:]

    num_time_steps = weights.shape[1]

    predictions = []

    for row in g:
        t = row[-1]
        closest_idx = int(np.round(t * (num_time_steps - 1)))
        w = weights[:, closest_idx]

        # Predict using the multi_obs predictors for the current row
        o_preds = np.array([multi_obs[i].predict(row.reshape(1, -1)) for i in range(len(multi_obs))]).flatten()

        # Combine the predictions using the weights for the current row
        prediction = np.dot(w, o_preds)
        predictions.append(prediction)

    return np.array(predictions)


def mm_plot_l2_tf(e, theta_true, theta_pred, multi_obs, lam):
    global prj_figs
    # Plot the L2 norm
    fig, ax1 = plot_l2_norm(e, theta_true, theta_pred)

    theta_true = theta_true.reshape(len(e), 1)

    tot = np.hstack((e, theta_true))
    final = tot[tot[:, 1]==1.0]
    xtr = np.unique(tot[:, 0])
    x = np.linspace(0, 1, 100)
    true = final[:, -1]

    Xobs = np.vstack((x, np.zeros_like(x), f2(np.ones_like(x)), f3(np.ones_like(x)), np.ones_like(x))).T
    pred = mm_predict(multi_obs, lam, Xobs)

    ax2 = fig.add_subplot(122)
    ax2.plot(xtr, true, marker="o", linestyle="None", alpha=1.0, linewidth=0.75, color='blue', label="true")
    ax2.plot(x, pred, linestyle='None', marker="X", linewidth=0.75, color='gold', label="mm_obs")

    colors = ['C3', 'lime', 'blue', 'aqua', 'm', 'darkred', 'k', 'yellow']

    for el in range(len(multi_obs)):
        ax2.plot(x, multi_obs[el].predict(Xobs), alpha=1.0, color=colors[el], linewidth=0.75, label=f"$obs_{el}$")

    ax2.set_xlabel(xlabel=r"Space x", fontsize=7)  # xlabel
    ax2.set_ylabel(ylabel=r"$\Theta$", fontsize=7)  # ylabel
    ax2.set_title(r"Prediction at tf", fontsize=7, weight='semibold')
    ax2.set_ylim(bottom=0.0)
    ax2.set_xlim(0, 1.01)
    ax2.legend()
    plt.yticks(fontsize=7)

    plt.grid()
    ax2.set_box_aspect(1)
    plt.savefig(f"{prj_figs}/l2_tf_lam{lam}.png", dpi=120)
    # plt.show()
    # plt.clf()
    plt.close()


def load_from_pickle(file_path):
    with open(file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)