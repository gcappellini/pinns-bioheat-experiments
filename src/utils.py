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

dde.config.set_random_seed(100)

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

properties = {
    "a1": None, "a2": None, "a3": None, "a4": None, "a5": None, "W": None
}

a1, a2, a3, a4, a5, a6 = 1.061375, 1.9125, 6.25e-05, 0.7, 15.0, 0.1666667

P0 = 1e+05
W = 0.45

f1, f2, f3 = [None]*3
upsilon = 500.0


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


# def get_properties(n):
#     global a1, a2, a3, a4, a5, a6
#     file_path = os.path.join(src_dir, 'simulations', f'data{n}.json')

#     # Open the file and load the JSON data
#     with open(file_path, 'r') as f:
#         data = json.load(f)

#     properties.update(data['Parameters'])
#     par = data['Parameters']
#     local_vars = locals()
#     for key in par:
#         if key in local_vars:
#             local_vars[key] = par[key]

#     a1, a2, a3, a4, a5, a6 = (
#         par["a1"], par["a2"], par["a3"], par["a4"], par["a5"], par["a6"]
#     )


def read_config():
    global run_logs
    filename = f"{run_logs}/config.json"
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            config = json.load(file)
    else:
        # Create default config if file doesn't exist
        config = create_default_config()
        write_config(config)
    return config


def create_default_config():
    # Define default configuration parameters
    network = {
        "activation": "elu", 
        "initial_weights_regularizer": True, 
        "initialization": "Glorot normal",
        "iterations": 30000,
        "LBFGS": False,
        "learning_rate": 0.0007607,
        "num_dense_layers": 5,
        "num_dense_nodes": 63,
        "output_injection_gain": 4,
        "resampling": True,
        "resampler_period": 100
    }
    return network

def write_config(config):
    global run_logs
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_config = {k: convert_to_serializable(v) for k, v in config.items()}

    filename = f"{run_logs}/config.json"
    with open(filename, 'w') as file:
        json.dump(serializable_config, file, indent=4)


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
    loss_bc1 = matrix[:, 1]    
    loss_ic = matrix[:, 2]

    fig = plt.figure(figsize=(6, 5))
    iters = losshistory.steps
    with sns.axes_style("darkgrid"):
        plt.clf()
        plt.plot(iters, loss_res, label=r'$\mathcal{L}_{res}$')
        plt.plot(iters, loss_bc1, label=r'$\mathcal{L}_{bc1}$')
        plt.plot(iters, loss_ic, label=r'$\mathcal{L}_{ic}$')
        plt.plot(iters, test, label='test loss')
        plt.plot(iters, train, label='train loss')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"{run_figs}/losses.png")
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
    global  a1, a2, a3, a4, a5, a6
    net = read_config()

    activation = net["activation"]
    initial_weights_regularizer = net["initial_weights_regularizer"]
    initialization = net["initialization"]
    learning_rate = net["learning_rate"]
    num_dense_layers = net["num_dense_layers"]
    num_dense_nodes = net["num_dense_nodes"]
    K = net["output_injection_gain"]


    def pde(x, theta, W):
        dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=3)
        dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)

        return (
            a1 * dtheta_tau
            - dtheta_xx + a2 * W * theta + a3 * P0 * torch.exp(-(1-x[:, 0:1])*a6)
        )
    
    def ic_obs(x):
        z = x[:, 0:1]
        y1 = 0
        y2 = x[:, 1:2]
        y3 = x[:, 2:3]

        b1 = a5 * y3 + (K - a5) * y2 - (2 + K) * a4

        return y1 + b1*z + a4*z**2


    def bc1_obs(x, theta, X):
        y2 = x[:, 1:2]
        y3 = x[:, 2:3]
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)

        return dtheta_x - a5 * (y3 - y2) - K * (y2 - theta)

    xmin = [0, 0, 0]
    xmax = [1, 1, 1]
    geom = dde.geometry.Cuboid(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)

    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_1, ic],
        num_domain=2560,
        num_boundary=200,
        num_initial=100,
        num_test=10000,
    )

    layer_size = [4] + [num_dense_nodes] * num_dense_layers + [1]
    net = dde.nn.FNN(layer_size, activation, initialization)

    net.apply_output_transform(output_transform)

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
    conf = read_config()
    mm = create_nbho()

    LBFGS = conf["LBFGS"]
    epochs = conf["iterations"]
    ini_w = conf["initial_weights_regularizer"]
    resampler = conf["resampling"]
    resampler_period = conf["resampler_period"]

    optim = "lbfgs" if LBFGS else "adam"
    iters = "*" if LBFGS else epochs

    # # Check if a trained model with the exact configuration already exists
    # trained_models = sorted(glob.glob(f"{run_models}/{optim}-{iters}.pt"))
    # if trained_models:
    #     mm.compile("L-BFGS") if LBFGS else None
    #     mm.restore(trained_models[0], verbose=0)
    #     return mm

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


def gen_testdata(n):
    data = np.loadtxt(f"{src_dir}/simulations/{n}/output_matlab.txt")
    x, t, exact, mm, sup, bol = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T, data[:, 3:4].T, data[:, 4:5].T, data[:, 5:6].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    y_mm = mm.flatten()[:, None]
    y_sup = sup.flatten()[:, None]
    y_bol = bol.flatten()[:, None]
    return X, y, y_mm, y_sup, y_bol


def gen_obsdata(n):
    global f2, f3
    g = np.hstack((gen_testdata(n)))
    instants = np.unique(g[:, 1])

    rows_1 = g[g[:, 0] == 1.0]


    y2 = rows_1[:, -2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    y3 = rows_1[:, -1].reshape(len(instants),)
    f3 = interp1d(instants, y3, kind='previous')

    
    # tm = 0.9957446808510638
    # if tau > tm:
    #     tau = tm

    Xobs = np.vstack((g[:, 0], f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    return Xobs


def plot_and_metrics(model, n_test):
    e, _, theta_true, _, _ = gen_testdata(n_test)
    g = gen_obsdata(n_test)

    theta_pred = model.predict(g)

    plot_comparison(e, theta_true, theta_pred)
    plot_l2_tf(e, theta_true, theta_pred, model)
    # plot_tf(e, theta_true, model)
    metrics = compute_metrics(theta_true, theta_pred)
    return metrics


def plot_comparison(e, theta_true, theta_pred, MObs=False):
    global run_figs, prj_figs

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

    if MObs:
        plt.savefig(f"{prj_figs}/comparison.png")

    else:
        plt.savefig(f"{run_figs}/comparison.png")

    plt.show()
    plt.close()
    plt.clf()


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
    ax1.plot(t, l2, alpha=1.0, linewidth=1.8, color='C0')
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
    fig, ax1 = plot_l2_norm(e, theta_true, theta_pred)

    tot = np.hstack((e, theta_true))
    final = tot[tot[:, 1]==1.0]
    xtr = np.unique(tot[:, 0])
    x = np.linspace(0, 1, 100)
    true = final[:, -1]

    Xobs = np.vstack((x, f2(np.ones_like(x)), f3(np.ones_like(x)), np.ones_like(x))).T
    pred = model.predict(Xobs)

    ax2 = fig.add_subplot(122)
    ax2.plot(xtr, true, marker="x", linestyle="None", alpha=1.0, color='C0', label="true")
    ax2.plot(x, pred, alpha=1.0, linewidth=1.8, color='C2', label="pred")

    ax2.set_xlabel(xlabel=r"Space x", fontsize=7)  # xlabel
    ax2.set_ylabel(ylabel=r"$\Theta$", fontsize=7)  # ylabel
    ax2.set_title(r"Prediction at tf", fontsize=7, weight='semibold')
    ax2.set_ylim(bottom=0.0)
    ax2.set_xlim(0, 1.01)
    ax2.legend()
    plt.yticks(fontsize=7)

    plt.grid()
    ax2.set_box_aspect(1)
    plt.savefig(f"{run_figs}/l2_tf.png")
    plt.show()
    plt.clf()


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


def single_observer(name_prj, name_run, n_test):
    # get_properties(n_test)
    wandb.init(
        project=name_prj, name=name_run,
        config=read_config()
    )
    mo = train_model()
    metrics = plot_and_metrics(mo, n_test)

    wandb.log(metrics)
    wandb.finish()
    return mo, metrics


def mm_observer(name_prj, n_test):
    global W, prj_logs
    # get_properties(n_test)
    # gen_obsdata(n_test)
    set_prj(name_prj)

    obs = np.array([1, 2, 3, 5, 6, 8, 9, 10])
    W_obs = np.dot(W, obs)

    n_obs = len(obs)

    # wandb.init(
    #     project=name_prj, name=name_run,
    #     config=read_config(name_run)
    # )

    multi_obs = []
    
    for j in range(n_obs):
        run = f"n{j}_W{W_obs[j]}"
        set_run(run)
        # config = read_config()
        W = W_obs[j]
        # write_config(config)
        model, _ = single_observer(name_prj, run, n_test)
        multi_obs.append(model)

    p0 = np.full((n_obs,), 1/n_obs)
    lem = [100, 500, 1000]

    for lam in lem:
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
        np.save(f'{prj_logs}/weights_lambda_{lam}.npy', weights)
        plot_weights(x, t, lam)
        metrics = mm_plot_and_metrics(multi_obs, n_test, lam)

    # wandb.log(metrics)
    # wandb.finish()
    # return mo, metrics


def mu(o, tau):
    global f2, f3
    xo = np.vstack((np.ones_like(tau), f2(tau), f3(tau), tau)).T
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
    # colors = ['C3', 'lime', 'blue', 'purple', 'aqua', 'lightskyblue', 'darkred', 'k']

    for i in range(x.shape[0]):
        # plt.plot(tauf * t, x[i], alpha=1.0, linewidth=1.8, color=colors[i], label=f"Weight $p_{i+1}$")
        plt.plot(t, x[i], alpha=1.0, linewidth=1.2, label=f"Weight $p_{i+1}$")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=0.0)

    ax1.set_xlabel(xlabel=r"Time t")  # xlabel
    ax1.set_ylabel(ylabel=r"Weights $p_j$")  # ylabel
    ax1.legend()
    ax1.set_title(f"Dynamic weights, $\lambda={lam}$", weight='semibold')
    plt.grid()
    plt.savefig(f"{prj_figs}/weights_lam_{lam}.png", dpi=150, bbox_inches='tight')

    plt.show()
    plt.clf()


def mm_plot_and_metrics(multi_obs, n_test, lam):
    e, _, theta_true = gen_testdata(n_test)
    g = gen_obsdata(n_test)

    theta_pred = mm_predict(multi_obs, lam, g).reshape(theta_true.shape)

    plot_comparison(e, theta_true, theta_pred, MObs=True)
    mm_plot_l2_tf(e, theta_true, theta_pred, multi_obs, lam)

    metrics = compute_metrics(theta_true, theta_pred)
    return metrics


def mm_predict(multi_obs, lam, g):
    global prj_logs
    a = np.load(f'{prj_logs}/weights_lambda_{lam}.npy', allow_pickle=True)
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

    Xobs = np.vstack((x, f2(np.ones_like(x)), f3(np.ones_like(x)), np.ones_like(x))).T
    pred = mm_predict(multi_obs, lam, Xobs)

    ax2 = fig.add_subplot(122)
    ax2.plot(xtr, true, marker="x", linestyle="None", alpha=1.0, color='C0', label="true")
    ax2.plot(x, pred, alpha=1.0, linewidth=1.8, color='C2', label="pred")

    ax2.set_xlabel(xlabel=r"Space x", fontsize=7)  # xlabel
    ax2.set_ylabel(ylabel=r"$\Theta$", fontsize=7)  # ylabel
    ax2.set_title(r"Prediction at tf", fontsize=7, weight='semibold')
    ax2.set_ylim(bottom=0.0)
    ax2.set_xlim(0, 1.01)
    ax2.legend()
    plt.yticks(fontsize=7)

    plt.grid()
    ax2.set_box_aspect(1)
    plt.savefig(f"{prj_figs}/l2_tf.png")
    plt.show()
    plt.clf()