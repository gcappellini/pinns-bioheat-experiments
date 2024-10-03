import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import deepxde as dde
import torch
import utils as uu
from omegaconf import OmegaConf

# Set up directories and random seed
dde.config.set_random_seed(200)
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)

models_dir = os.path.join(git_dir, "models")
os.makedirs(models_dir, exist_ok=True)

def plot_generic(x, y, title, xlabel, ylabel, legend_labels=None, log_scale=False, 
                 size=(6, 5), filename=None, colors=None, linestyles=None):
    """
    Create a generic 2D plot with support for multiple lines, colors, and linestyles.

    :param x: x-axis data (1D array or list of arrays)
    :param y: y-axis data (list of arrays or 2D array for multiple lines)
    :param title: plot title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param legend_labels: labels for the legend (must match the number of y lines)
    :param log_scale: set y-axis to logarithmic scale if True
    :param size: figure size
    :param filename: path to save the plot (optional)
    :param colors: list of colors for each line (optional)
    :param linestyles: list of linestyles for each line (optional)
    """
    fig, ax = plt.subplots(figsize=size)

    # Plot each line with its corresponding x, y values, color, and linestyle
    for i, (xi, yi) in enumerate(zip(x, y)):
        label = legend_labels[i] if legend_labels else None
        color = colors[i] if colors else None  # Use provided colors or default
        linestyle = linestyles[i] if linestyles else '-'  # Default to solid line

        ax.plot(xi, yi, label=label, color=color, linestyle=linestyle)

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

    # Save and close the figure if filename is provided
    save_and_close(fig, filename)


def plot_generic_3d(XY, Z1, Z2, xlabel, ylabel, zlabel, col_titles,  filename=None):
  
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
def plot_loss_components(losshistory, nam):
    # Prepare the loss data
    loss_train = np.array(losshistory.loss_train)
    loss_test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    train = loss_train.sum(axis=1).ravel()

    # Extract individual loss components
    loss_res, loss_bc0, loss_bc1, loss_ic = loss_train[:, 0], loss_train[:, 1], loss_train[:, 2], loss_train[:, 3]
    
    # Combine all loss components into a list/2D array for plotting
    y_values = [loss_res, loss_bc0, loss_bc1, loss_ic, loss_test, train]
    
    # Labels for the legend
    legend_labels = [r'$\mathcal{L}_{res}$', r'$\mathcal{L}_{bc0}$', r'$\mathcal{L}_{bc1}$', r'$\mathcal{L}_{ic}$', 'test loss', 'train loss']
    
    # Get iterations (x-axis)
    iters = losshistory.steps
    loss_terms = np.array(y_values)
    iterations = np.array([iters]*len(y_values))
    conf = OmegaConf.load(f"{src_dir}/config.yaml")
    colors = conf.plot.colors.losses

    # Call the generic plotting function
    plot_generic(
        x=iterations,
        y=loss_terms,
        title="Loss Components",
        xlabel="iterations",
        ylabel="loss",
        legend_labels=legend_labels,
        log_scale=True,  # We want a log scale on the y-axis
        filename=f"{models_dir}/losses_{nam}.png",
        colors=colors
    )


def plot_weights(weights, t, run_figs, conf, gt=False):
    lam = conf.model_parameters.lam
    n_obs = conf.model_parameters.n_obs

    # Prepare the labels for each weight line
    legend_labels = [f"Weight $p_{i}$" for i in range(weights.shape[0])]

    # Define the title with the lambda value
    title = f"Dynamic weights, λ={lam}"
    a = t.reshape(len(t),)
    times = np.full_like(weights, a)

    colors = uu.get_obs_colors(conf)
    rescale = conf.plot.rescale
    _, xlabel, _ = uu.get_scaled_labels(rescale) 
    
    # Call the generic plotting function
    plot_generic(
        x=times,                       # Time data for the x-axis
        y=weights,                 # Weights data (each row is a separate line)
        title=title,               # Plot title with dynamic lambda value
        xlabel=xlabel,      # x-axis label
        ylabel=r"Weights $p_j$",    # y-axis label
        legend_labels=legend_labels, # Labels for each weight
        size=(6, 5),               # Figure size
        colors=colors,
        filename=f"{run_figs}/weights_lam_{lam}_{'matlab' if gt else 'pinns'}_{n_obs}obs.png"  # Filename to save the plot
    )


def plot_mu(mus, t, run_figs, conf, gt=False):
    n_obs = conf.model_parameters.n_obs
    # Prepare the labels for each line based on the number of columns in `mus`
    legend_labels = [f"$e_{i}$" for i in range(mus.shape[1])]
    
    # Define the title for the plot
    title = "Observation errors"
    times = [t]*mus.shape[1]
    colors = uu.get_obs_colors(conf)
    rescale = conf.plot.rescale
    times_plot = uu.rescale_t(times) if rescale else times     
    # Call the generic plotting function
    plot_generic(
        x=times_plot,                       # Time data for the x-axis
        y=mus.T,                   # Transpose mus to get lines for each observation error
        title=title,               # Plot title
        xlabel=r"Time $\tau$",      # x-axis label
        ylabel=r"Error $\mu$",             # y-axis label
        legend_labels=legend_labels, # Labels for each observation error
        size=(6, 5),               # Figure size
        colors=colors,
        filename=f"{run_figs}/obs_error_{'matlab' if gt else 'pinns'}_{n_obs}obs.png"  # Filename to save the plot
    )


def plot_l2(xobs, theta_true, model, number, folder, MultiObs=False):
    """
    Plot L2 norm of prediction errors for true and predicted values.
    
    :param xobs: Input observations (depth and time).
    :param theta_true: True theta values.
    :param model: A single model or a list of models if MultiObs is True.
    :param number: Identifier for the observation.
    :param folder: Directory to save the figure.
    :param MultiObs: If True, use multiple models for predictions.
    """
    # Reshape the inputs
    e = xobs[:, 0:2].reshape(len(xobs), 2)
    theta_true = theta_true.reshape(len(e), 1)

    f = OmegaConf.load(f'{folder}/config.yaml')
    lam = f.model_parameters.lam

    # Prepare for L2 norm computation
    l2 = []

    # Extract unique time values
    t = np.unique(e[:, 1])
    t_filtered = t[t > 0.000]  # Filter out small time values

    # Store corresponding time values for each model's predictions
    x_vals = [t_filtered]  # Initialize with time values for the combined prediction or single prediction

    if MultiObs:
        # Combine predictions using mm_predict
        combined_pred = uu.mm_predict(model, lam, xobs, folder)

        # Calculate L2 error for combined prediction
        tot_combined = np.hstack((e, theta_true, combined_pred.reshape(len(e), 1)))
        l2_combined = []
        for el in t_filtered:
            df = tot_combined[tot_combined[:, 1] == el]
            l2_combined.append(dde.metrics.l2_relative_error(df[:, 2], df[:, 3]))
        l2.append(l2_combined)  # Store the combined L2 error

        # Calculate L2 error for each individual model
        for i, individual_model in enumerate(model):  # Assuming `model.models` holds individual models
            theta_pred = individual_model.predict(xobs).reshape(len(e), 1)
            tot_individual = np.hstack((e, theta_true, theta_pred))
            l2_individual = []
            for el in t_filtered:
                df = tot_individual[tot_individual[:, 1] == el]
                l2_individual.append(dde.metrics.l2_relative_error(df[:, 2], df[:, 3]))
            l2.append(l2_individual)  # Store the individual model L2 error
            x_vals.append(t_filtered)  # Add corresponding x values for each individual model

    else:
        # Single model prediction
        theta_pred = model.predict(xobs).reshape(len(e), 1)
        tot = np.hstack((e, theta_true, theta_pred))
        for el in t_filtered:
            df = tot[tot[:, 1] == el]
            l2.append(dde.metrics.l2_relative_error(df[:, 2], df[:, 3]))
        x_vals = [t_filtered]  # Just a single time series for x values

    # Prepare labels for the legend
    legend_labels = ['MultiObs Pred'] if MultiObs else ['Predicted']

    # Add individual model labels if MultiObs is True
    if MultiObs:
        for i in range(len(model)):
            legend_labels.append(f'Obs {i}')
    ll2 = np.array(l2)
    times = [t[:-1]]*len(ll2)
    # Call the generic plotting function
    plot_generic(
        x=times,   # Provide time values for each line (either one for each model or just one for single prediction)
        y=ll2,       # Multiple L2 error lines to plot
        title="Prediction error norm",
        xlabel=r"Time $\tau$",
        ylabel=r"$L^2$ norm",
        legend_labels=legend_labels,  # Labels for the legend
        size=(6, 5),
        filename=f"{folder}/l2_{'mm_obs' if MultiObs else f'obs{number}'}.png"
    )


def plot_tf(e, theta_true, model, number, prj_figs, MultiObs=False):
    """
    Plot true values and predicted values (single or multi-observer model).
    
    :param e: 2D array for depth (X) and time (tau).
    :param theta_true: True theta values.
    :param model: The model used for predictions.
    :param number: Identifier for the observation.
    :param prj_figs: Directory to save the figure.
    :param MultiObs: If True, plot multiple model predictions and a weighted average.
    """
    # Reshape inputs
    e = e.reshape(len(e), 2)
    theta_true = theta_true.reshape(len(e), 1)

    # Prepare true values for final time (tau = 1)
    tot = np.hstack((e, theta_true))
    final = tot[tot[:, 1] == 1]  # Select rows where time equals 1
    xtr = np.unique(tot[:, 0])   # Depth values for true data
    true = final[:, -1]          # True values at final time

    # Generate X values for prediction
    # x = np.linspace(0, 1, 100)  # Depth values for prediction
    x=xtr
    Xobs = np.vstack((x, uu.f1(np.full_like(x, 0.9944)), uu.f2(np.full_like(x, 0.9944)), uu.f3(np.full_like(x, 0.9944)), np.ones_like(x))).T

    conf = OmegaConf.load(f'{prj_figs}/config.yaml')
    lam = conf.model_parameters.lam
    n_obs = conf.model_parameters.n_obs
    obs_colors = uu.get_obs_colors(conf)
    true_color, mm_obs_color = uu.get_sys_mm_colors(conf)
    obs_linestyles = uu.get_obs_linestyle(conf)
    true_linestyle, mm_obs_linestyle = uu.get_sys_mm_linestyle(conf)

    if MultiObs:
        # Combined prediction using multi-observer model
        multi_pred = uu.mm_predict(model, lam, Xobs, prj_figs)

        # Generate individual predictions from each model in the ensemble
        individual_preds = [m.predict(Xobs) for m in model]  # Assuming model.models holds individual models
        
        # Stack all predictions (true values + individual predictions + combined prediction)
        all_preds = [true] + individual_preds + [multi_pred]
        
        # x values: use xtr for true data, and 'x' for predictions
            # xxtr = xtr.reshape(len(xtr), 1)
        x_vals = [xtr] + Xobs[:, 0:1]*n_obs + Xobs[:, 0:1]

        # Generate corresponding legend labels
        legend_labels = ['True'] + [f'Obs {i}' for i in range(len(individual_preds))] + ['MultiObs Pred']

        colors = [true_color] + obs_colors + [mm_obs_color]
        linestyles = [true_linestyle] + obs_linestyles + [mm_obs_linestyle]
    else:
        # Single model prediction
        pred = model.predict(Xobs)
        
        # Only two lines to plot: true and single prediction
        all_preds = [true, pred]
        x_vals = [xtr, x]  # xtr for true, x for predicted
        legend_labels = ['True', f'Obs {number}']
        colors = [true_color] + [obs_colors[number]]
        linestyles = [true_linestyle] + [obs_linestyles[number]]

    rescale = conf.plot.rescale
    xlabel, _, ylabel = uu.get_scaled_labels(rescale)

    x_vals = np.array(x_vals, dtype=float)
    x_vals_plot = np.hstack((uu.rescale_x(x_vals[:, 0:1]), uu.rescale_time(x_vals[:, 1:2]))) if rescale else x_vals
    all_preds_plot = uu.rescale_t(all_preds) if rescale else all_preds

    # Call the generic plotting function
    plot_generic(
        x=x_vals_plot,  # Different x values for true and predicted
        y=all_preds_plot,  # List of true and predicted lines
        title="Prediction at final time",
        xlabel=xlabel,
        ylabel=ylabel,
        legend_labels=legend_labels,  # Labels for the legend
        size=(6, 5),
        filename=f"{prj_figs}/tf_{'mm_obs' if MultiObs else f'obs{number}'}.png",
        colors = colors,
        linestyles=linestyles
    )


def plot_tf_matlab(e, theta_true, theta_observers, theta_mmobs, conf, prj_figs):
    """
    Plot true values and predicted values (single or multi-observer model).
    
    :param e: 2D array for depth (X) and time (tau).
    :param theta_true: True theta values.
    :param model: The model used for predictions.
    :param number: Identifier for the observation.
    :param prj_figs: Directory to save the figure.
    :param MultiObs: If True, plot multiple model predictions and a weighted average.
    """

    # Reshape inputs
    e = e.reshape(len(e), 2)
    theta_true = theta_true.reshape(len(e), 1)
    theta_mmobs = theta_mmobs.reshape(len(e), 1)

    # Prepare true values for final time (tau = 1)
    tot = np.hstack((e, theta_true, theta_observers, theta_mmobs))
    final = tot[tot[:, 1] == 1]  # Select rows where time equals 1
    xtr = np.unique(tot[:, 0])   # Depth values for true data
    true = final[:, 2]          # True values at final time
    observers = final[:, 3:-1]
    mmobs = final[:, -1]

    true_reshaped = true.reshape(len(xtr), 1)
    mmobs_reshaped = mmobs.reshape(len(xtr), 1)
    all_preds = np.hstack((observers, true_reshaped, mmobs_reshaped))
    # x values: use xtr for true data, and 'x' for predictions
    xxtr = xtr.reshape(len(xtr), 1)
    x_vals = np.full_like(all_preds, xxtr)
    number = conf.model_parameters.n_obs

    # Generate corresponding legend labels
    legend_labels = [f'Obs {i}' for i in range(number)] + ['True'] + ['MultiObs']

    obs_colors = uu.get_obs_colors(conf)
    system_color, mm_obs_color = uu.get_sys_mm_colors(conf)
    obs_linestyles = uu.get_obs_linestyle(conf)
    system_linestyle, mm_obs_linestyle = uu.get_sys_mm_linestyle(conf)

    colors = obs_colors + [system_color] + [mm_obs_color]
    linestyles = obs_linestyles + [system_linestyle] + [mm_obs_linestyle]
    rescale = conf.plot.rescale
    xlabel, _, ylabel = uu.get_scaled_labels(rescale)

    x_plot = uu.rescale_x(x_vals.T) if rescale else x_vals.T
    y_plot = uu.rescale_t(all_preds.T) if rescale else all_preds.T


    # Call the generic plotting function
    plot_generic(
        x=x_plot,  # Different x values for true and predicted
        y=y_plot,  # List of true and predicted lines
        title="Prediction at final time",
        xlabel=xlabel,
        ylabel=ylabel,
        legend_labels=legend_labels,  # Labels for the legend
        size=(6, 5),
        colors=colors,
        filename=f"{prj_figs}/tf_mmobs_matlab_{number}obs.png",
        linestyles=linestyles
    )


def plot_comparison_3d(e, t_true, t_pred, run_figs, gt=False):
    """
    Refactor the plot_comparison function to use plot_generic_3d for 3D comparisons.
    
    :param e: 2D array for X and Y data.
    :param t_true: True values reshaped for 3D plotting.
    :param t_pred: Predicted values reshaped for 3D plotting.
    :param run_figs: Directory to save the plot.
    """
    # Determine the unique points in X and Y dimensions
    la = len(np.unique(e[:, 0]))  # Number of unique X points
    le = len(np.unique(e[:, 1]))  # Number of unique Y points
    
    # Reshape the true and predicted theta values to match the 2D grid
    theta_true = t_true.reshape(le, la)
    theta_pred = t_pred.reshape(le, la)

    # Column titles for each subplot
    col_titles = ["System", "MultiObserver", "Error"]

    fname = f"{run_figs}/comparison_3d_matlab.png" if gt else f"{run_figs}/comparison_3d_pinns.png"
    conf = OmegaConf.load(f"{run_figs}/config.yaml")
    rescale = conf.plot.rescale

    theta_true_plot = uu.rescale_t(theta_true) if rescale else theta_true
    theta_pred_plot = uu.rescale_t(theta_pred) if rescale else theta_pred
    xlabel, ylabel, zlabel = uu.get_scaled_labels(rescale)
    # Call plot_generic_3d with the data
    plot_generic_3d(
        e,                       # 2D array containing X and Y coordinates
        theta_true_plot,              # Surface 1: true values (Z1)
        theta_pred_plot,              # Surface 2: predicted values (Z2)
        xlabel, ylabel, zlabel,
        col_titles,      # Titles for each subplot
        fname
    )


def plot_timeseries_with_predictions(df, y1_pred, gt1_pred, gt2_pred, y2_pred, prj_figs):
    conf = OmegaConf.load(f"{prj_figs}/config.yaml")
    # Prepare x-axis data (time in minutes)
    time_in_minutes = df['t'] / 60
    
    # Prepare y-axis data (ground truth and predicted values)
    y_data = [
        df['y1'], df['gt1'], df['gt2'], df['y2'],  # Ground truth lines
        y1_pred, gt1_pred, gt2_pred, y2_pred       # Predicted lines
    ]

    # Labels for the legend (corresponding to each line in y_data)
    legend_labels = ['y1 (True)', 'gt1 (True)', 'gt2 (True)', 'y2 (True)', 
                     'y1 (Pred)', 'gt1 (Pred)', 'gt2 (Pred)', 'y2 (Pred)']
    colors_points = conf.plot.colors.measuring_points
    colors = colors_points * 2
    linestyles=["-", "-", "-", "-", "--", "--", "--", "--"]

    times = [time_in_minutes]*len(y_data)
    # Call the generic plotting function
    plot_generic(
        x=times,        # Time data
        y=y_data,                 # All y data (ground truth + predictions)
        title="Cooling Experiment",
        xlabel="Time (min)",
        ylabel="Temperature (°C)",
        legend_labels=legend_labels,
        colors=colors,
        linestyles=linestyles,
        size=(12, 6),
        filename=f"{prj_figs}/timeseries_with_predictions.png"
    )