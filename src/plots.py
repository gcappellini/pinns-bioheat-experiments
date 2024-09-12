import deepxde as dde
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import wandb
import json
import pickle
import pandas as pd
import utils as uu
from common import set_run


dde.config.set_random_seed(200)

# device = torch.device("cpu")
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)

models = os.path.join(git_dir, "models")
os.makedirs(models, exist_ok=True)


def plot_loss_components(losshistory):
    global models
    prop = uu.read_json("properties.json")
    hash = uu.generate_config_hash(prop)
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


def plot_weights(weights, t, run_figs, gt=False):
    param = uu.read_json("parameters.json")
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


def plot_mu(mus, t, run_figs, gt=False):

    true_mus = uu.compute_mu()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    colors = ['C3', 'lime', 'blue', 'aqua', 'm', 'darkred', 'k', 'yellow']
    if gt:
        for i in range(mus.shape[0]):
            plt.plot(t, true_mus[i], alpha=0.6, linestyle="-.", color=colors[i], label=f"$e_{i}$ matlab")
            plt.plot(t, mus[i], alpha=1.0, linewidth=1.0, color=colors[i], label=f"$e_{i}$")
    else:
        for i in range(mus.shape[0]):
            plt.plot(t, mus[i], alpha=1.0, linewidth=1.0, color=colors[i], label=f"$e_{i}$")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=0.0)

    ax1.set_xlabel(xlabel=r"Time t")  # xlabel
    ax1.set_ylabel(ylabel=r"Error")  # ylabel
    ax1.legend()
    ax1.set_title(r"Observation errors", weight='semibold')
    plt.grid()

    plt.savefig(f"{run_figs}/obs_error.png", dpi=120, bbox_inches='tight')

    # plt.show()
    plt.close()
    # plt.clf()


def check_obs(e, theta_true, theta_pred, number, run_figs):

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
    plt.savefig(f"{run_figs}/check_obs_{number}.png", dpi=120)

    # plt.show()
    plt.close()
    # plt.clf()

    m = uu.compute_metrics(theta_true, theta_pred)
    with open(f"{run_figs}/metrics_obs_{number}.json", 'w') as json_file:
        json.dump(m, json_file, indent=4) 


def plot_comparison(e, t_true, t_pred, run_figs):

    la = len(np.unique(e[:, 0]))
    le = len(np.unique(e[:, 1]))

    theta_true = t_true.reshape(le, la)
    theta_pred = t_pred.reshape(le, la)

    # Predictions
    fig = plt.figure(3, figsize=(9, 4))

    col_titles = ['Measured', 'MM Observer', 'Error']
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


def plot_l2_tf(e, theta_true, theta_pred, model, number, run_figs):

    e, theta_true, theta_pred = e.reshape((len(e), 2)), theta_true.reshape((len(e), 1)), theta_pred.reshape((len(e), 1))

    fig, ax1 = plot_l2_norm(e, theta_true, theta_pred)

    tot = np.hstack((e, theta_true))
    final = tot[tot[:, 1]==1]
    xtr = np.unique(tot[:, 0])
    x = np.linspace(0, 1, 100)
    true = final[:, -1]

    Xobs = np.vstack((x, uu.f1(np.ones_like(x)), uu.f2(np.ones_like(x)), uu.f3(np.ones_like(x)), np.ones_like(x))).T
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
    plt.savefig(f"{run_figs}/l2_tf_obs{number}.png", dpi=120)
    
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


def mm_plot_l2_tf(e, theta_true, theta_pred, multi_obs, lam, run_figs):
    # Plot the L2 norm
    fig, ax1 = plot_l2_norm(e, theta_true, theta_pred)

    theta_true = theta_true.reshape(len(e), 1)

    tot = np.hstack((e, theta_true))
    final = tot[tot[:, 1]==1.0]
    xtr = np.unique(tot[:, 0])
    x = np.linspace(0, 1, 100)
    true = final[:, -1]

    Xobs = np.vstack((x, np.zeros_like(x), uu.f2(np.ones_like(x)), uu.f3(np.ones_like(x)), np.ones_like(x))).T
    pred = uu.mm_predict(multi_obs, lam, Xobs)

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
    plt.savefig(f"{run_figs}/l2_tf_lam{lam}.png", dpi=120)
    # plt.show()
    # plt.clf()
    plt.close()


def plot_and_metrics(model):

    o = uu.import_testdata()
    e, theta_true = o[:, 0:2], o[:, -2]
    g = uu.import_obsdata()

    theta_pred = model.predict(g)

    plot_comparison(e, theta_true, theta_pred)
    # check_obs(e, theta_obs, theta_pred)
    plot_l2_tf(e, theta_true, theta_pred, model)
    # plot_tf(e, theta_true, model)
    metrics = uu.compute_metrics(theta_true, theta_pred)
    return metrics





def mm_plot_and_metrics(multi_obs, lam, n):
    e, theta_true = uu.gen_testdata(n)
    g = uu.gen_obsdata(n)
    # a = import_testdata()
    # e = a[:, 0:2]
    # theta_true = a[:, 2]
    # g = import_obsdata()

    theta_pred = uu.mm_predict(multi_obs, lam, g).reshape(theta_true.shape)

    plot_comparison(e, theta_true, theta_pred, MObs=True)
    mm_plot_l2_tf(e, theta_true, theta_pred, multi_obs, lam)

    metrics = uu.compute_metrics(theta_true, theta_pred)
    return metrics
