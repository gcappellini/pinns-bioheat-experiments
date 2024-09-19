import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import deepxde as dde
import torch
import utils as uu
import common as co

# Set up directories and random seed
dde.config.set_random_seed(200)
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)

models_dir = os.path.join(git_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# Load parameters and properties
param = co.read_json(f"{src_dir}/parameters.json")
prop = co.read_json(f"{src_dir}/properties.json")
config_hash = co.generate_config_hash(prop)
lam = param["lambda"]

def plot_generic(x, y, title, xlabel, ylabel, legend_labels=None, log_scale=False, size=(6, 5), filename=None):
    """
    Create a generic 2D plot with support for multiple lines.

    :param x: x-axis data (1D array)
    :param y: y-axis data (can be a list of arrays or a 2D array for multiple lines)
    :param title: plot title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param legend_labels: labels for the legend (must match the number of y lines)
    :param log_scale: set y-axis to logarithmic scale if True
    :param size: figure size
    :param filename: path to save the plot (optional)
    """
    fig, ax = plt.subplots(figsize=size)

    # Check if y is a 2D array or list of arrays
    if isinstance(y, (list, np.ndarray)) and np.ndim(y) > 1:
        # Plot each line in y with its corresponding legend label
        for i, yi in enumerate(y):
            label = legend_labels[i] if legend_labels else None
            ax.plot(x, yi, label=label)
    else:
        # If y is a single line, plot it directly
        ax.plot(x, y, label=legend_labels)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add legend if labels are provided
    if legend_labels:
        ax.legend()

    # Set y-axis to log scale if specified
    if log_scale:
        ax.set_yscale('log')

    ax.grid(True)
    save_and_close(fig, filename)

def plot_generic_3d(XY, Z1, Z2, col_titles,  filename=None):
    xlabel="X"
    ylabel=r"$\tau$"
    zlabel=r"$\theta$"
    
    fig = plt.figure(3, figsize=(9, 4))
    surfaces = [
        [Z1, Z2,
            np.abs(Z1 - Z2)]
    ]
    grid = plt.GridSpec(1, 3)
    for col in range(3):
        ax = fig.add_subplot(grid[0, col], projection='3d')
        configure_subplot(ax, XY, surfaces[0][col], xlabel, ylabel, zlabel)

        # Set column titles
        ax.set_title(col_titles[col], fontsize=8, y=.96, weight='semibold')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.15)
    save_and_close(fig, filename)


# Helper functions for common plotting tasks
def create_plot(title, xlabel, ylabel, size=(6, 5)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def save_and_close(fig, filename):
    fig.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


def configure_subplot(ax, XS, surface, xlabel, ylabel, zlabel):
    la = len(np.unique(XS[:, 0:1]))
    le = len(np.unique(XS[:, 1:]))
    X = XS[:, 0].reshape(le, la)
    T = XS[:, 1].reshape(le, la)

    ax.plot_surface(X, T, surface, cmap='inferno', alpha=0.8)
    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.dist = 10
    ax.view_init(20, -120)
    ax.set_xlabel(xlabel, fontsize=7, labelpad=-1)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=-1)
    ax.set_zlabel(zlabel, fontsize=7, labelpad=-4)


# Main plot functions
def plot_loss_components(losshistory):
    loss_train = np.array(losshistory.loss_train)
    loss_test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    train = loss_train.sum(axis=1).ravel()
    loss_res, loss_bc0, loss_bc1, loss_ic = loss_train[:, 0], loss_train[:, 1], loss_train[:, 2], loss_train[:, 3]

    fig, ax = create_plot("Loss Components", "iterations", "loss")
    iters = losshistory.steps
    sns.set_style("darkgrid")

    ax.plot(iters, loss_res, label=r'$\mathcal{L}_{res}$')
    ax.plot(iters, loss_bc0, label=r'$\mathcal{L}_{bc0}$')
    ax.plot(iters, loss_bc1, label=r'$\mathcal{L}_{bc1}$')
    ax.plot(iters, loss_ic, label=r'$\mathcal{L}_{ic}$')
    ax.plot(iters, loss_test, label='test loss')
    ax.plot(iters, train, label='train loss')

    ax.set_yscale('log')
    ax.legend(ncol=2)
    save_and_close(fig, f"{models_dir}/losses_{config_hash}.png")


def plot_weights(weights, t, run_figs, gt=False):
    fig, ax = create_plot(f"Dynamic weights, λ={lam}", r"Time $\tau$", r"Weights $p_j$")
    colors = ['C3', 'lime', 'blue', 'aqua', 'm', 'darkred', 'k', 'yellow']

    for i, color in enumerate(colors[:weights.shape[0]]):
        ax.plot(t, weights[i], color=color, label=f"Weight $p_{i}$")

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    save_and_close(fig, f"{run_figs}/weights_lam_{lam}_{'matlab' if gt else 'pinns'}.png")


def plot_mu(mus, t, run_figs, gt=False):
    fig, ax = create_plot("Observation errors", r"Time $\tau$", "Error")
    colors = ['C3', 'lime', 'blue', 'aqua', 'm', 'darkred', 'k', 'yellow']

    for i, color in enumerate(colors[:mus.shape[1]]):
        ax.plot(t, mus[:, i], color=color, label=f"$e_{i}$")

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    save_and_close(fig, f"{run_figs}/obs_error_{'matlab' if gt else 'pinns'}.png")


def plot_l2(e, theta_true, theta_pred, number, folder, MultiObs=False):
    e, theta_true, theta_pred = e.reshape(len(e), 2), theta_true.reshape(len(e), 1), theta_pred.reshape(len(e), 1)
    t = np.unique(e[:, 1])
    t_filtered = t[t > 0.000]
    l2 = []

    theta_true = theta_true.reshape(len(e), 1)
    theta_pred = theta_pred.reshape(len(e), 1)
    tot = np.hstack((e, theta_true, theta_pred))

    for el in t_filtered:
        df = tot[tot[:, 1] == el]
        l2.append(dde.metrics.l2_relative_error(df[:, 2], df[:, 3]))

    fig, ax = create_plot("Prediction error norm", r"Time $\tau$", r"$L^2$ norm")
    ax.plot(t_filtered, l2, color='C0')
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(0, 1.01)
    ax.grid()

    save_and_close(fig, f"{folder}/l2_{'mm_obs' if MultiObs else f'obs{number}'}.png")


def plot_tf(e, theta_true, theta_pred, model, number, prj_figs, MultiObs=False):
    e, theta_true, theta_pred = e.reshape(len(e), 2), theta_true.reshape(len(e), 1), theta_pred.reshape(len(e), 1)

    tot = np.hstack((e, theta_true))
    final = tot[tot[:, 1] == 1]
    xtr = np.unique(tot[:, 0])
    x = np.linspace(0, 1, 100)
    true = final[:, -1]

    Xobs = np.vstack((x, uu.f1(np.ones_like(x)), uu.f2(np.ones_like(x)), uu.f3(np.ones_like(x)), np.ones_like(x))).T
    pred = uu.mm_predict(model, lam, Xobs, prj_figs) if MultiObs else model.predict(Xobs)

    fig, ax2 = create_plot("Prediction at final time", r"Depth $X$", r"$\theta$")
    ax2.plot(xtr, true, marker="x", linestyle="None", color='C0', label="true")
    ax2.plot(x, pred, color='C2', label="pred")

    ax2.set_xlim(0, 1.01)
    ax2.set_ylim(bottom=0.0)
    ax2.legend()

    save_and_close(fig, f"{prj_figs}/tf_{'mm_obs' if MultiObs else f'obs{number}'}.png")


def plot_comparison(e, t_true, t_pred, run_figs):
    la = len(np.unique(e[:, 0]))
    le = len(np.unique(e[:, 1]))
    theta_true = t_true.reshape(le, la)
    theta_pred = t_pred.reshape(le, la)

    fig = plt.figure(figsize=(9, 4))
    col_titles = ['Measured', 'MM Observer', 'Error']
    surfaces = [theta_true, theta_pred, np.abs(theta_true - theta_pred)]
    
    for col in range(3):
        ax = fig.add_subplot(1, 3, col, projection='3d')
        configure_subplot(ax, e, surfaces[col])
        ax.set_title(col_titles[col], fontsize=8, weight='semibold')

    plt.subplots_adjust(wspace=0.15)
    save_and_close(fig, f"{run_figs}/comparison.png")


def plot_timeseries_with_predictions(df, y1_pred, gt1_pred, gt2_pred, y2_pred, prj_figs):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['t'] / 60, df['y1'], linestyle="--", alpha=0.8)
    ax.plot(df['t'] / 60, df['gt1'], linestyle="--", alpha=0.8)
    ax.plot(df['t'] / 60, df['gt2'], linestyle="--", alpha=0.8)
    ax.plot(df['t'] / 60, df['y2'], linestyle="--", alpha=0.8)

    ax.plot(df['t'] / 60, y1_pred, label='y1', color="C0", linewidth=0.7)
    ax.plot(df['t'] / 60, gt1_pred, label='gt1', color="C1", linewidth=0.7)
    ax.plot(df['t'] / 60, gt2_pred, label='gt2', color="C2", linewidth=0.7)
    ax.plot(df['t'] / 60, y2_pred, label='y2', color="C3", linewidth=0.7)

    ax.legend()
    ax.set_title("Cooling Experiment", fontweight='bold')
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)

    save_and_close(fig, f"{prj_figs}/comparison.png")