import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import deepxde as dde
import torch
import yaml

# Load configuration file (config.yaml)
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Define paths
models_dir = "path_to_models_directory"

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

def plot_generic(x, y, title, xlabel, ylabel, legend_labels=None, log_scale=False, size=(6, 5), filename=None):
    fig, ax = plt.subplots(figsize=size)

    # Plot each line with its corresponding x and y values
    for i, (xi, yi) in enumerate(zip(x, y)):
        label = legend_labels[i] if legend_labels else None
        linestyle = config['plot']['observers']['linestyle']
        color = config['plot']['observers']['colors'][i % len(config['plot']['observers']['colors'])]
        alpha = config['plot']['observers']['alpha']
        ax.plot(xi, yi, label=label, linestyle=linestyle, color=color, alpha=alpha)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if legend_labels:
        ax.legend()

    if log_scale:
        ax.set_yscale('log')

    ax.grid(True)
    save_and_close(fig, filename)

# Main plot functions
def plot_loss_components(losshistory):
    loss_train = np.array(losshistory.loss_train)
    loss_test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    train = loss_train.sum(axis=1).ravel()

    loss_res, loss_bc0, loss_bc1, loss_ic = loss_train[:, 0], loss_train[:, 1], loss_train[:, 2], loss_train[:, 3]
    y_values = [loss_res, loss_bc0, loss_bc1, loss_ic, loss_test, train]

    legend_labels = [r'$\mathcal{L}_{res}$', r'$\mathcal{L}_{bc0}$', r'$\mathcal{L}_{bc1}$', r'$\mathcal{L}_{ic}$', 'test loss', 'train loss']

    iters = losshistory.steps
    iterations = np.array([iters] * len(y_values))

    plot_generic(
        x=iterations,
        y=y_values,
        title="Loss Components",
        xlabel="iterations",
        ylabel="loss",
        legend_labels=legend_labels,
        log_scale=True,
        filename=f"{models_dir}/losses.png"
    )

def plot_weights(weights, t, run_figs, lam, gt=False):
    legend_labels = [f"Weight $p_{i}$" for i in range(weights.shape[0])]
    title = f"Dynamic weights, λ={lam}"
    times = np.full_like(weights, t)

    plot_generic(
        x=times,
        y=weights,
        title=title,
        xlabel=r"Time $\tau$",
        ylabel=r"Weights $p_j$",
        legend_labels=legend_labels,
        filename=f"{run_figs}/weights_lam_{lam}_{'matlab' if gt else 'pinns'}.png"
    )

# Additional plotting functions will follow the same structure, using values from `config.yaml` where needed.