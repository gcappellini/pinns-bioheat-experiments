import deepxde as dde
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import wandb
from scipy.interpolate import interp1d
from scipy import integrate
import pickle
import pandas as pd
import coeff_calc as cc
import hashlib
import plots as pp
import common as co
from omegaconf import OmegaConf
import matlab.engine

dde.config.set_random_seed(200)

dev = torch.device("cpu")
# dev = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)


models = os.path.join(git_dir, "models")
os.makedirs(models, exist_ok=True)

f1, f2, f3 = [None]*3


def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(0)
    return losshistory.loss_train[0]


def load_from_pickle(file_path):
    with open(file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)
    

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

def create_nbho(run_figs):
    config = OmegaConf.load(f'{run_figs}/config.yaml')

    activation = config.model_properties.activation
    initial_weights_regularizer = config.model_properties.initial_weights_regularizer
    initialization = config.model_properties.initialization
    learning_rate = config.model_properties.learning_rate
    num_dense_layers = config.model_properties.num_dense_layers
    num_dense_nodes = config.model_properties.num_dense_nodes
    w_res, w_bc0, w_bc1, w_ic = config.model_properties.w_res, config.model_properties.w_bc0, config.model_properties.w_bc1, config.model_properties.w_ic
    num_domain, num_boundary, num_initial, num_test = config.model_properties.num_domain, config.model_properties.num_boundary, config.model_properties.num_initial, config.model_properties.num_test

    a1 = cc.a1
    a2 = cc.a2
    a3 = cc.a3
    K = config.model_properties.K
    delta = config.model_properties.delta
    W = config.model_properties.W

    # Twater = config.model_properties.Twater
    # Ty20 = config.model_properties.Ty20
    # theta_w, theta_y20 = scale_t(Twater), scale_t(Ty20)


    def pde(x, theta):
        dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=4)
        dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)

        return (
            a1 * dtheta_tau
            - dtheta_xx + W * a2 * theta
        )
    
    def ic_obs(x):
        z = x[:, 0:1]

        # theta_y20 = x[:, 2:3]
        # theta_w = x[:, 3:4]
        conf = OmegaConf.load(f"{src_dir}/config.yaml")
        theta_y20 = scale_t(conf.model_properties.Ty20)
        theta_w = scale_t(conf.model_properties.Twater)


        c = theta_y20
        b = a3*(theta_w-theta_y20) + theta_y20
        a = delta

        return (1-z)*(a*z**2+b*z+c)

    def bc1_obs(x, theta, X):

        return theta - x[:, 1:2]

    def bc0_obs(x, theta, X):
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)

        return - dtheta_x - a3 * (x[:, 3:4] - x[:, 2:3]) - K * (x[:, 2:3] - theta)
        # return dtheta_x - a3 * (x[:, 3:4] - x[:, 2:3]) - K * (x[:, 2:3] - theta)

    xmin = [0, 0, 0, 0]
    xmax = [1, 0.2, 1, 1]
    geom = dde.geometry.Hypercube(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.OperatorBC(geomtime, bc0_obs, boundary_0)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)

    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        lambda x, theta: pde(x, theta),
        [bc_0, bc_1, ic],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
    )

    layer_size = [5] + [num_dense_nodes] * num_dense_layers + [1]
    net = dde.nn.FNN(layer_size, activation, initialization)

    # net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    if initial_weights_regularizer:
        initial_losses = get_initial_loss(model)
        loss_weights = len(initial_losses)/ initial_losses
        model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    else:
        loss_weights = [w_res, w_bc0, w_bc1, w_ic]
        model.compile("adam", lr=learning_rate, loss_weights=loss_weights)

    return model


def train_model(run_figs):
    global models
    config = OmegaConf.load(f'{run_figs}/config.yaml')
    config_hash = co.generate_config_hash(config.model_properties)
    n = config.model_properties.iterations
    model_path = os.path.join(models, f"model_{config_hash}.pt-{n}.pt")

    mm = create_nbho(run_figs)

    if os.path.exists(model_path):
        # Model exists, load it
        print(f"Loading model from {model_path}")
        mm.restore(model_path, device=torch.device(dev), verbose=0)
        return mm
    
    LBFGS = config.model_properties.LBFGS
    ini_w = config.model_properties.initial_weights_regularizer
    resampler = config.model_properties.resampling
    resampler_period = config.model_properties.resampler_period

    callbacks = [dde.callbacks.PDEPointResampler(period=resampler_period)] if resampler else []

    if LBFGS:
        if ini_w:
            initial_losses = get_initial_loss(mm)
            loss_weights = len(initial_losses) / initial_losses
            mm.compile("L-BFGS", loss_weights=loss_weights)
        else:
            mm.compile("L-BFGS")
        
        losshistory, _ = train_and_save_model(mm, callbacks, run_figs)
    else:
        losshistory, _ = train_and_save_model(mm, callbacks, run_figs)

    pp.plot_loss_components(losshistory, config_hash)
    return mm


def train_and_save_model(model, callbacks, run_figs):
    global models

    conf = OmegaConf.load(f'{run_figs}/config.yaml')
    config_hash = co.generate_config_hash(conf.model_properties)
    model_path = os.path.join(models, f"model_{config_hash}.pt")

    display_every = 1000
    losshistory, train_state = model.train(
        iterations=conf.model_properties.iterations,
        callbacks=callbacks,
        model_save_path=model_path,
        display_every=display_every
    )
    confi_path = os.path.join(models, f"config_{config_hash}.yaml")
    OmegaConf.save(conf, confi_path)

    return losshistory, train_state


def gen_testdata(n):
    if n==8:
        data = np.loadtxt(f"{src_dir}/output_matlab_{n}Obs.txt")
        x, t, sys, obs, mmobs = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T, data[:, 3:11], data[:, 11:12].T
    if n==3:
        data = np.loadtxt(f"{src_dir}/output_matlab_{n}Obs.txt")
        x, t, sys, obs, mmobs = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T, data[:, 3:6], data[:, 6:7].T       
    X = np.vstack((x, t)).T
    y_sys = sys.flatten()[:, None]
    y_mmobs = mmobs.flatten()[:, None]

    return X, y_sys, obs, y_mmobs


def load_weights(n):
    if n==8:
        data = np.loadtxt(f"{src_dir}/weights_matlab_{n}Obs.txt")
        t, weights = data[:, 0:1], data[:, 1:9].T
    if n==3:
        data = np.loadtxt(f"{src_dir}/weights_matlab_{n}Obs.txt")
        t, weights = data[:, 0:1], data[:, 1:4].T
    return t, np.array(weights)


def gen_obsdata(n):
    global f1, f2, f3
    g = np.hstack((gen_testdata(n)))
    instants = np.unique(g[:, 1])
    
    rows_1 = g[g[:, 0] == 1.0]
    rows_0 = g[g[:, 0] == 0.0]

    y1 = rows_1[:, 2].reshape(len(instants),)
    f1 = interp1d(instants, y1, kind='previous')

    y2 = rows_0[:, 2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    theta_w = scale_t(properties.model_parameters.Twater)
    y3 = np.full_like(y2, theta_w)
    f3 = interp1d(instants, y3, kind='previous')

    Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    return Xobs

def gen_obs_y2(n):
    global f1, f2, f3
    g = np.hstack((gen_testdata(n)))
    instants = np.unique(g[:, 1])
    
    rows_1 = g[g[:, 0] == 1.0]
    rows_0 = g[g[:, 0] == 0.0]

    y1 = rows_1[:, 2].reshape(len(instants),)
    f1 = interp1d(instants, y1, kind='previous')

    y2 = rows_0[:, 2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    theta_w = scale_t(properties.model_parameters.Twater)
    y3 = np.full_like(y2, theta_w)
    f3 = interp1d(instants, y3, kind='previous')

    Xobs = np.vstack((np.zeros_like(instants), f1(instants), f2(instants), f3(instants), instants)).T
    return Xobs, f2(instants)


def gen_obs_y1(n):
    global f1, f2, f3
    g = np.hstack((gen_testdata(n)))
    instants = np.unique(g[:, 1])
    
    rows_1 = g[g[:, 0] == 1.0]
    rows_0 = g[g[:, 0] == 0.0]

    y1 = rows_1[:, 2].reshape(len(instants),)
    f1 = interp1d(instants, y1, kind='previous')

    y2 = rows_0[:, 2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    theta_w = scale_t(properties.model_parameters.Twater)
    y3 = np.full_like(y2, theta_w)
    f3 = interp1d(instants, y3, kind='previous')

    Xobs = np.vstack((np.ones_like(instants), f1(instants), f2(instants), f3(instants), instants)).T
    return Xobs, f1(instants)

def load_from_pickle(file_path):
    with open(file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)
    
def scale_t(t):
    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    Troom = properties.model_properties.Troom
    Tmax = properties.model_properties.Tmax
    k = (t - Troom) / (Tmax - Troom)

    return round(k, 4)

def rescale_t(theta):
    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    Troom = properties.model_properties.Troom
    Tmax = properties.model_properties.Tmax
    theta = np.array(theta, dtype=float)
    j = Troom + (Tmax - Troom)*theta

    return np.round(j, 2)

def rescale_x(X):
    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    L0 = properties.model_properties.L0
    j = X*L0

    return np.round(j, 4)

def rescale_time(tau):
    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    tauf = properties.model_properties.tauf
    j = tau*tauf

    return np.round(j, 0)

def scale_time(t):
    properties = OmegaConf.load(f"{src_dir}/config.yaml")
    tauf = properties.model_properties.tauf
    j = t/tauf

    return np.round(j, 4)

def get_tc_positions():
    daa = OmegaConf.load(f"{src_dir}/config.yaml")
    L0 = daa.model_properties.L0
    x_y2 = 0
    x_gt2 = 0.01/L0
    x_gt1 = 0.04/L0
    x_y1 = 1

    return [x_y2, x_gt2, x_gt1, x_y1] 

def import_testdata(name):
    df = load_from_pickle(f"{src_dir}/{name}.pkl")

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


def import_obsdata(nam):
    global f1, f2, f3
    g = import_testdata(nam)
    instants = np.unique(g[:, 1])

    positions = get_tc_positions()

    rows_1 = g[g[:, 0] == positions[-1]]
    rows_0 = g[g[:, 0] == positions[0]]

    y1 = rows_1[:, -2].reshape(len(instants),)
    f1 = interp1d(instants, y1, kind='previous')

    y2 = rows_0[:, -2].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    # y3 = rows_0[:, -1].reshape(len(instants),)
    y3 = np.zeros_like(y2)
    f3 = interp1d(instants, y3, kind='previous')

    Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    return Xobs


def mm_observer(config):

    n_obs = config.model_parameters.n_obs

    if n_obs==8:
        W0, W1, W2, W3, W4, W5, W6, W7 = config.model_parameters.W0, config.model_parameters.W1, config.model_parameters.W2, config.model_parameters.W3, config.model_parameters.W4, config.model_parameters.W5, config.model_parameters.W6, config.model_parameters.W7
        obs = np.array([W0, W1, W2, W3, W4, W5, W6, W7])
    if n_obs==3:
        W0, W1, W2 = config.model_parameters.W0, config.model_parameters.W4, config.model_parameters.W7
        obs = np.array([W0, W1, W2])        




    multi_obs = []
    
    for j in range(n_obs):
        perf = obs[j]
        config.model_properties.W = float(perf)
        run_figs = co.set_run(f"obs_{j}")
        OmegaConf.save(config, f"{run_figs}/config.yaml") 

        model = train_model(run_figs)
        multi_obs.append(model)


    return multi_obs

def check_mm_obs(multi_obs, x_obs, X, y_sys, conf, prj_figs):
    run_figs = co.set_run(f"mm_obs")
    conf.model_properties.W = None
    OmegaConf.save(conf, f"{run_figs}/config.yaml") 
    # Solve IVP and plot weights
    solve_ivp_and_plot(multi_obs, run_figs, conf, x_obs, X, y_sys)


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


def compute_mu(n_obs):

    g = np.hstack((gen_testdata(n_obs)))
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



def mm_predict(multi_obs, lam, obs_grid, prj_figs):
    a = np.load(f'{prj_figs}/weights_lam_{lam}.npy', allow_pickle=True)
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


def test_observer(model, run_figs, X, x_obs, y_obs, number):
        obs = f"Obs {number}"

        conf = OmegaConf.load(f'{run_figs}/config.yaml')
        n_obs = conf.model_parameters.n_obs

        
        # Model prediction
        y_pred = model.predict(x_obs)
        la = len(np.unique(X[:, 0]))
        le = len(np.unique(X[:, 1]))

        true = y_obs[:, number].reshape(le, la)
        pred = y_pred.reshape(le, la)
        y2_true = true[:, 0]
        y1_true = true[:, -1]

        Xob_y2, _ = gen_obs_y2(n_obs)
        Xob_y1, _ = gen_obs_y1(n_obs)
        y2_pred, y1_pred = model.predict(Xob_y2), model.predict(Xob_y1)

        t = np.unique(Xob_y2[:, -1])

        y = np.vstack([y2_true, y2_pred.reshape(y2_true.shape), y1_true, y1_pred.reshape(y2_true.shape)])
        legend_labels = [r'$\hat{\theta}_{true}(0, \tau)$', r'$\hat{\theta}_{pred}(0, \tau)$', r'$\hat{\theta}_{true}(1, \tau)$', r'$\hat{\theta}_{pred}(1, \tau)$']

        # Check Model prediction
        pp.plot_generic_3d(X[:, 0:2], pred, true, ["PINNs", "Matlab", "Error"], filename=f"{run_figs}/comparison_3d_{obs}")
        pp.plot_generic(t, y, "Comparison at the boundary", r"Time ($\tau$)", r"Theta ($\theta$)", legend_labels, filename=f"{run_figs}/comparison_outputs_{obs}")
        # Compute and log errors
        errors = compute_metrics(y_obs[:, number], y_pred)
        return errors


def check_observers_and_wandb_upload(multi_obs, x_obs, X, y_sys, conf, output_dir):
    """
    Check observers and optionally upload results to wandb.
    """
    run_wandb = conf.experiment.run_wandb
    exp_type = conf.experiment.type
    name = conf.experiment.name
    for el in range(len(multi_obs)):

        run_figs = os.path.join(output_dir, f"obs_{el}")

        if run_wandb:
            aa = OmegaConf.load(f"{run_figs}/config.yaml")
            print(f"Initializing wandb for observer {el}...")
            wandb.init(project=name, name=f"obs_{el}", config=aa)
        
        pred = multi_obs[el].predict(x_obs)
        
        # pp.plot_l2(x_obs, y_sys, multi_obs[el], el, run_figs)
        pp.plot_tf(X, y_sys, multi_obs[el], el, run_figs)
        pp.plot_comparison_3d(X, y_sys, pred, run_figs)
        
        if exp_type == "simulation":
            pp.plot_comparison_3d(X[:, 0:2], y_sys, pred, run_figs)

        metrics = compute_metrics(y_sys, pred)
 
        if run_wandb:
            wandb.log(metrics)
            wandb.finish()


def get_scaled_labels(rescale):
    xlabel=r"$x \, (m)$" if rescale else "X"
    ylabel=r"$t \, (s)$" if rescale else r"$\tau$"
    zlabel=r"$T \, (^{\circ}C)$" if rescale else r"$\theta$"
    return xlabel, ylabel, zlabel



def get_obs_colors(conf):
    total_obs_colors = conf.plot.colors.observers
    number = conf.model_parameters.n_obs
    obs_colors = total_obs_colors[:number]
    return obs_colors

def get_sys_mm_colors(conf):
    system_color = conf.plot.colors.system
    mm_obs_color = conf.plot.colors.mm_obs
    return system_color, mm_obs_color

def get_obs_linestyle(conf):
    total_obs_linestyles = conf.plot.linestyles.observers
    number = conf.model_parameters.n_obs
    obs_colors = total_obs_linestyles[:number]
    return obs_colors

def get_sys_mm_linestyle(conf):
    system_linestyle = conf.plot.linestyles.system
    mm_obs_linestyle = conf.plot.linestyles.mm_obs
    return system_linestyle, mm_obs_linestyle


def solve_ivp_and_plot(multi_obs, fold, conf, x_obs, X, y_sys):
    """
    Solve the IVP for observer weights and plot the results.
    """
    n_obs = conf.model_parameters.n_obs
    lam = conf.model_parameters.lam
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
    
    np.save(f'{fold}/weights_lam_{lam}.npy', weights)
    pp.plot_weights(weights[1:], weights[0], fold, conf)
    
    # Model prediction
    y_pred = mm_predict(multi_obs, lam, x_obs, fold)
    t = np.unique(X[:, 1:2])
    mus = mu(multi_obs, t)

    # if run_wandb:
    #     print(f"Initializing wandb for multi observer ...")
    #     wandb.init(project= str, name=f"mm_obs")

    metrics = compute_metrics(y_sys, y_pred)
    
    # if run_wandb:
    #     wandb.log(metrics)
    #     wandb.finish()

    pp.plot_mu(mus, t, fold, conf)
    # pp.plot_l2(x_obs, y_sys, multi_obs, 0, fold, MultiObs=True)
    pp.plot_tf(X, y_sys, multi_obs, 0, fold, MultiObs=True)
    pp.plot_comparison_3d(X[:, 0:2], y_sys, y_pred, fold)




def run_matlab_ground_truth(src_dir, prj_figs, conf, run_matlab):
    """
    Optionally run MATLAB ground truth.
    """
    n_obs = conf.model_parameters.n_obs
    
    if run_matlab:
        print("Running MATLAB ground truth calculation...")
        eng = matlab.engine.start_matlab()
        eng.cd(src_dir, nargout=0)
        eng.BioHeat(nargout=0)
        eng.quit()

        X, y_sys, y_observers, y_mmobs = gen_testdata(n_obs)
        t = np.unique(X[:, 1])

        mu = compute_mu(n_obs)
        pp.plot_mu(mu, t, prj_figs, conf, gt=True)

        t, weights = load_weights(n_obs)
        pp.plot_weights(weights, t, prj_figs, conf, gt=True)
        pp.plot_tf_matlab(X, y_sys, y_observers, y_mmobs, conf, prj_figs)
        pp.plot_comparison_3d(X, y_sys, y_mmobs, prj_figs, gt= True)


        print("MATLAB ground truth completed.")
    else:
        print("Skipping MATLAB ground truth calculation.")