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

dde.config.set_random_seed(200)

# device = torch.device("cpu")
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)


models = os.path.join(tests_dir, "models")
os.makedirs(models, exist_ok=True)

figures = os.path.join(tests_dir, "figures")
os.makedirs(figures, exist_ok=True)

logs = os.path.join(tests_dir, "logs")
os.makedirs(logs, exist_ok=True)

prj_figs, prj_models, prj_logs, run_figs, run_models, run_logs = [None]*6


f1, f2, f3 = [None]*3



def set_prj(prj):
    global prj_figs, prj_models, prj_logs

    prj_logs = os.path.join(logs, prj)
    os.makedirs(prj_logs, exist_ok=True)

    prj_models = os.path.join(models, prj)
    os.makedirs(prj_models, exist_ok=True)

    prj_figs = os.path.join(figures, prj)
    os.makedirs(prj_figs, exist_ok=True)

    return prj_figs, prj_models, prj_logs


def set_run(run):
    global prj_figs, prj_models, prj_logs, run_figs, run_models, run_logs

    run_logs = os.path.join(prj_logs, run)
    os.makedirs(run_logs, exist_ok=True)

    run_models = os.path.join(prj_models, run)
    os.makedirs(run_models, exist_ok=True)

    run_figs = os.path.join(prj_figs, run)
    os.makedirs(run_figs, exist_ok=True)

    return run_figs, run_models, run_logs



def read_json(filename):
    filepath = f"{run_logs}/{filename}"
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
    else:
        if filename == "config.json":
            data = create_default_config()
        elif filename == "properties.json":
            data = create_default_properties()
        write_json(data, filename)
    return data


def create_default_config():
    network = {
        "activation": "tanh", 
        "initial_weights_regularizer": True, 
        "initialization": "Glorot normal",
        "iterations": 30000,
        "LBFGS": False,
        "learning_rate": 0.001,
        "num_dense_layers": 4,
        "num_dense_nodes": 100,
        "resampling": True,
        "resampler_period": 100
    }
    return network


def create_default_properties():
    properties = {
        "a1": cc.a1,
        "a2": cc.a2,
        "a3": cc.a3,
        "a4": cc.a4,
        "a5": cc.a5,
        "a6": cc.a6,
        "lam": 10.0,
        "output_injection_gain": 15.0,
        "upsilon": 5.0,
    }
    return properties

def write_json(data, filename):
    global run_logs
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_data = {k: convert_to_serializable(v) for k, v in data.items()}

    filepath = f"{run_logs}/{filename}"
    with open(filepath, 'w') as file:
        json.dump(serializable_data, file, indent=4)


def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(0)
    return losshistory.loss_train[0]



def plot_loss_components(losshistory):
    global run_figs
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
        plt.savefig(f"{run_figs}/losses.png", dpi=120)
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

    K = properties["output_injection_gain"]
    a1 = properties["a1"]
    a2 = properties["a2"]
    a3 = properties["a3"]
    a4 = properties["a4"]
    a5 = properties["a5"]
    a6 = properties["a6"]


    def pde(x, theta):
        dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=4)
        dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)

        return (
            a1 * dtheta_tau
            - dtheta_xx + a2 * theta - a3 * torch.exp(-(1-x[:, 0:1])*a6)
        )
    
    def ic_obs(x):
        z = x[:, 0:1]
        y1_0 = x[:, 1:2]
        y2_0 = x[:, 2:3]
        y3_0 = x[:, 3:4]

        b1 = (a5*y3_0+(K-a5)*y2_0-(2+K)*a4)/(1+K)

        return y1_0 + b1*z + a4*z**2

    def bc0_obs(x, theta, X):

        return theta - x[:, 1:2]

    def bc1_obs(x, theta, X):
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)

        return dtheta_x - a5 * (x[:, 3:4] - x[:, 2:3]) - K * (x[:, 2:3] - theta)

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
    global run_models
    conf = read_json("config.json")
    mm = create_nbho()

    LBFGS = conf["LBFGS"]
    epochs = conf["iterations"]
    ini_w = conf["initial_weights_regularizer"]
    resampler = conf["resampling"]
    resampler_period = conf["resampler_period"]

    optim = "lbfgs" if LBFGS else "adam"
    iters = "*" if LBFGS else epochs

    # Check if a trained model with the exact configuration already exists
    trained_models = sorted(glob.glob(f"{run_models}/{optim}-{iters}.pt"))
    if trained_models:
        mm.compile("L-BFGS") if LBFGS else None
        mm.restore(trained_models[0], verbose=0)
        return mm

    callbacks = [dde.callbacks.PDEPointResampler(period=resampler_period)] if resampler else []

    if LBFGS:
        # Attempt to restore from a previously trained Adam model if exists
        adam_models = sorted(glob.glob(f"{run_models}/adam-{epochs}.pt"))
        if adam_models:
            mm.restore(adam_models[0], verbose=0)
        else:
            losshistory, train_state = train_and_save_model(mm, epochs, callbacks, "adam")
        
        if ini_w:
            initial_losses = get_initial_loss(mm)
            loss_weights = len(initial_losses) / initial_losses
            mm.compile("L-BFGS", loss_weights=loss_weights)
        else:
            mm.compile("L-BFGS")
        
        losshistory, train_state = train_and_save_model(mm, epochs, callbacks, "lbfgs")
    else:
        losshistory, train_state = train_and_save_model(mm, epochs, callbacks, "adam")

    plot_loss_components(losshistory)
    return mm


def train_and_save_model(model, iterations, callbacks, optimizer_name):
    global run_models
    display_every = 1000
    losshistory, train_state = model.train(
        iterations=iterations,
        callbacks=callbacks,
        model_save_path=f"{run_models}/{optimizer_name}",
        display_every=display_every
    )
    return losshistory, train_state


def gen_testdata():
    data = np.loadtxt(f"{src_dir}/data/simulations/output_matlab.txt")
    x, t, exact, obs1, mm, sup, bol = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T, data[:, 3:4].T, data[:, 4:5].T, data[:, 5:6].T, data[:, 6:7].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    y_obs1 = obs1.flatten()[:, None]
    y_mm = mm.flatten()[:, None]
    y_sup = sup.flatten()[:, None]
    y_bol = bol.flatten()[:, None]
    return X, y, y_obs1, y_mm, y_sup, y_bol


def gen_obsdata():
    global f1, f2, f3
    g = np.hstack((gen_testdata()))
    instants = np.unique(g[:, 1])

    rows_1 = g[g[:, 0] == 1.0]

    def f1(j):
        return np.zeros_like(j)


    y2 = rows_1[:, -2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    y3 = rows_1[:, -1].reshape(len(instants),)
    f3 = interp1d(instants, y3, kind='previous')

    Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    return Xobs

def import_testdata(n):
    path = f"{src_dir}/data/measurements/vessel/{n}.pkl"
    df = load_from_pickle(path)
    x_tcs = np.linspace(0, 1, num=8).round(4)
    x_y1 = x_tcs[0]
    x_gt1 = x_tcs[3]
    x_gt2 = x_tcs[6]
    x_y2 = x_tcs[7]

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


def import_obsdata(n):
    global f1, f2, f3
    g = import_testdata(n)
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
    e, theta_true, theta_obs, _, _, _ = gen_testdata()
    g = gen_obsdata()

    # o = import_testdata(n_test)
    # e, theta_true = o[:, 0:2], o[:, 2]
    # g = import_obsdata(n_test)

    theta_pred = model.predict(g)

    plot_comparison(e, theta_true, theta_pred)
    check_obs(e, theta_obs, theta_pred)
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
    t_filtered = t[t > 0.02]

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
    ax.view_init(20, -210)

    # Set axis labels
    ax.set_xlabel('Depth', fontsize=7, labelpad=-1)
    ax.set_ylabel('Time', fontsize=7, labelpad=-1)
    ax.set_zlabel('Theta', fontsize=7, labelpad=-4)


def single_observer(name_prj, name_run):
    set_prj(name_prj)
    set_run(name_run)
    config = read_json("config.json")
    properties = read_json("properties.json")
    combined_config = {**config, **properties}

    # wandb.init(
    #     project=name_prj, name=name_run,
    #     config=combined_config
    # )
    mo = train_model()
    metrics = plot_and_metrics(mo)
    # wandb.log(metrics)
    # wandb.finish()

    return mo, metrics


def mm_observer(name_prj):
    global prj_logs
    # get_properties(n_test)

    set_prj(name_prj)

    obs = np.array([1, 2, 3, 5, 6, 8, 9, 10])
    a2_obs = np.dot(cc.a2, obs).round(4)

    n_obs = len(obs)

    # wandb.init(
    #     project=name_prj, name=name_run,
    #     config=read_config(name_run)
    # )

    multi_obs = []
    
    for j in range(n_obs):
        run = f"n{j}_W{a2_obs[j]}"
        set_run(run)
        # config = read_config()
        a2_new = a2_obs[j]
        properties = read_json("properties.json")
        lam = properties["lam"]
        properties["a2"] = a2_new
        write_json(properties, "properties.json")
        model, _ = single_observer(name_prj, run)
        multi_obs.append(model)

    p0 = np.full((n_obs,), 1/n_obs)

    def f(t, p):
        a = mu(multi_obs, t)
        e = np.exp(-1*a)
        d = np.inner(p, e)
        f = []
        for el in range(len(p)):
            ccc = - lam * (1-(e[el]/d))*p[el]
            f.append(ccc)
        return np.array(f)


    sol = integrate.solve_ivp(f, (0, 1), p0, t_eval=np.linspace(0, 1, 100))
    x = sol.y
    t = sol.t
    weights = np.zeros((sol.y.shape[0]+1, sol.y.shape[1]))
    weights[0] = sol.t
    weights[1:] = sol.y
    np.save(f'{prj_logs}/weights_lam_{lam}.npy', weights)
    plot_weights(x, t, lam)
    plot_mu(multi_obs, t)
    metrics = mm_plot_and_metrics(multi_obs, lam)

    # wandb.log(metrics)
    # wandb.finish()
    # return mo, metrics


def mu(o, tau):
    global f1, f2, f3
    net = read_json("properties.json")
    K = net["output_injection_gain"]
    upsilon = net["upsilon"]

    xo = np.vstack((np.ones_like(tau), f1(tau), f2(tau), f3(tau), tau)).T
    muu = []
    for el in o:
        oss = el.predict(xo)
        scrt = upsilon*np.abs(oss-f2(tau))**2
        muu.append(scrt)
    muu = np.array(muu).reshape(len(muu),)
    return muu


def plot_weights(x, t, lam):
    global prj_figs
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    colors = ['C3', 'lime', 'blue', 'aqua', 'purple', 'darkred', 'k', 'yellow']

    for i in range(x.shape[0]):
        # plt.plot(tauf * t, x[i], alpha=1.0, linewidth=1.8, color=colors[i], label=f"Weight $p_{i+1}$")
        plt.plot(t, x[i], alpha=1.0, linewidth=1.0, color=colors[i], label=f"Weight $p_{i}$")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=0.0)

    ax1.set_xlabel(xlabel=r"Time t")  # xlabel
    ax1.set_ylabel(ylabel=r"Weights $p_j$")  # ylabel
    ax1.legend()
    ax1.set_title(r"Dynamic weights, $\lambda=$"f"{lam}", weight='semibold')
    plt.grid()
    plt.savefig(f"{prj_figs}/weights_lam_{lam}.png", dpi=120, bbox_inches='tight')

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
    colors = ['C3', 'lime', 'blue', 'aqua', 'purple', 'darkred', 'k', 'yellow']
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


def mm_plot_and_metrics(multi_obs, lam):
    e, theta_true, _, _, _, _ = gen_testdata()
    g = gen_obsdata()

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

    tot = np.hstack((e, theta_true))
    final = tot[tot[:, 1]==1.0]
    xtr = np.unique(tot[:, 0])
    x = np.linspace(0, 1, 100)
    true = final[:, -1]

    Xobs = np.vstack((x, np.zeros_like(x), f2(np.ones_like(x)), f3(np.ones_like(x)), np.ones_like(x))).T
    pred = mm_predict(multi_obs, lam, Xobs)

    ax2 = fig.add_subplot(122)
    ax2.plot(xtr, true, marker="o", linestyle="None", alpha=1.0, linewidth=0.75, color='blue', label="true", markevery=6)
    ax2.plot(x, pred, linestyle='None', marker="X", linewidth=0.75, color='gold', label="mm_obs", markevery=6)

    colors = ['C3', 'lime', 'blue', 'aqua', 'purple', 'darkred', 'k', 'yellow']

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