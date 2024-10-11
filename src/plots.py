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
                 size=(6, 5), filename=None, colors=None, linestyles=None, markers=None,
                 linewidths=None, markersizes=None, alphas=None):
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

    # If colors is a single string, wrap it in a list
    if isinstance(colors, str):
        colors = [colors]
    # If linestyles is a single string, wrap it in a list
    if isinstance(linestyles, str):
        linestyles = [linestyles]

    # Plot each line with its corresponding x, y values, color, and linestyle
    for i, (xi, yi) in enumerate(zip(x, y)):
        label = legend_labels[i] if legend_labels else None
        color = colors[i] if colors else None  # Use provided colors or default
        linestyle = linestyles[i] if linestyles else '-'  # Default to solid line
        marker = markers[i] if markers else None
        linewidth = linewidths[i] if linewidths else 1.2
        markersize = markersizes[i] if markersizes else 4
        alpha=alphas[i] if alphas else 1

        ax.plot(xi, yi, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth, markersize=markersize, alpha=alpha)

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
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
  
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


# def plot_weights(weights, t, run_figs, conf, gt=False):
def plot_weights(weights, t, run_figs, conf1, gt=False):
    lam = conf1.model_parameters.lam
    n_obs = conf1.model_parameters.n_obs

    # Prepare the labels for each weight line
    legend_labels = [f"Weight $p_{i}$" for i in range(weights.shape[0])]

    # Define the title with the lambda value
    title = f"Dynamic weights, λ={lam}"
    a = t.reshape(len(t),)
    times = np.full_like(weights, a)

    conf = OmegaConf.load(f"{src_dir}/config.yaml")
    colors = uu.get_obs_colors(conf)
    linestyles = uu.get_obs_linestyles(conf)
    rescale = conf.plot.rescale
    _, xlabel, _ = uu.get_scaled_labels(rescale) 
    times_plot = uu.rescale_time(times) if rescale else times
    
    # Call the generic plotting function
    plot_generic(
        x=times_plot,                       # Time data for the x-axis
        y=weights,                 # Weights data (each row is a separate line)
        title=title,               # Plot title with dynamic lambda value
        xlabel=xlabel,      # x-axis label
        ylabel=r"Weights $p_j$",    # y-axis label
        legend_labels=legend_labels, # Labels for each weight
        size=(6, 5),               # Figure size
        colors=colors,
        linestyles=linestyles,
        filename=f"{run_figs}/weights_lam_{lam}_{'matlab' if gt else 'pinns'}_{n_obs}obs.png"  # Filename to save the plot
    )


def plot_mu(mus, t, run_figs, gt=False):
    conf = OmegaConf.load(f"{src_dir}/config.yaml")
    n_obs = conf.model_parameters.n_obs
    # Prepare the labels for each line based on the number of columns in `mus`
    legend_labels = [f"$e_{i}$" for i in range(mus.shape[1])]
    
    t = t.reshape(len(t), 1)
    # Define the title for the plot
    title = "Observation errors"
    times = np.full_like(mus, t)
    colors = uu.get_obs_colors(conf)
    linestyles = uu.get_obs_linestyles(conf)
    rescale = conf.plot.rescale
    times_plot = uu.rescale_time(times) if rescale else times  
    _, xlabel, _ = uu.get_scaled_labels(rescale) 

    # if gt:
    times_plot = times_plot.T
    mus = mus.T
    # Call the generic plotting function
    plot_generic(
        x=np.array(times_plot),                       # Time data for the x-axis
        y=np.array(mus),                   # Transpose mus to get lines for each observation error
        title=title,               # Plot title
        xlabel=xlabel,      # x-axis label
        ylabel=r"Error $\mu$",             # y-axis label
        legend_labels=legend_labels, # Labels for each observation error
        size=(6, 5),               # Figure size
        colors=colors,
        linestyles=linestyles,
        filename=f"{run_figs}/obs_error_{'matlab' if gt else 'pinns'}_{n_obs}obs.png"  # Filename to save the plot
    )


def plot_l2(tot_true, tot_pred, number, folder, MultiObs=False):
    """
    Plot L2 norm of prediction errors for true and predicted values.
    
    :param xobs: Input observations (depth and time).
    :param theta_true: True theta values.
    :param model: A single model or a list of models if MultiObs is True.
    :param number: Identifier for the observation.
    :param folder: Directory to save the figure.
    :param MultiObs: If True, use multiple models for predictions.
    """

    matching = uu.extract_matching(tot_true, tot_pred)
    e = matching[:, :2].reshape(len(matching), 2)

    theta_true = matching[:, 2].reshape(len(matching), 1)

    conf = OmegaConf.load(f'{folder}/config.yaml')

    n_obs = conf.model_parameters.n_obs
    obs_colors = uu.get_obs_colors(conf)
    obs_linestyles = uu.get_obs_linestyles(conf)
    _, mm_obs_color = uu.get_sys_mm_colors(conf)
    _, mm_obs_linestyle = uu.get_sys_mm_linestyle(conf)


    if MultiObs:
        pred = matching[:, -1].reshape(len(matching[:, -1]), 1)

        # combined_pred = uu.mm_predict(model, lam, xobs, folder)
        l2 = uu.calculate_l2(e, theta_true, pred)
        # ll2 = l2.reshape(len(l2), 1)

        # Calculate L2 error for each individual model
        l2_individual = []
        for i in range(n_obs):  # Assuming `model.models` holds individual models
            theta_pred = matching[:, 3+i].reshape(len(matching), 1)
            l2_individual.append(uu.calculate_l2(e, theta_true, theta_pred))

        l2_individual_obs = np.array(l2_individual).T
        l2 = l2.reshape(len(l2), 1)
        ll2 = np.hstack((l2, l2_individual_obs))

    else:

        pred = matching[:, 3+number].reshape(len(matching), 1)
        l2_individual = uu.calculate_l2(e, theta_true, pred)
        ll2 = l2_individual.reshape(len(l2_individual), 1)

    # Prepare labels for the legend
    legend_labels = ['MultiObs'] + [f'Obs {i}' for i in range(n_obs)] if MultiObs else [f'Obs {number}']
    # legend_labels = ['MultiObs'] if MultiObs else [f'Obs {number}']
    colors = [mm_obs_color] + obs_colors if MultiObs else [obs_colors[number]]
    # colors = [mm_obs_color] if MultiObs else [obs_colors[number]]
    linestyles = [mm_obs_linestyle] + obs_linestyles if MultiObs else [obs_linestyles[number]]
    # linestyles = [mm_obs_linestyle] if MultiObs else [obs_linestyles[number]]


    rescale = conf.plot.rescale
    _, xlabel, _ = uu.get_scaled_labels(rescale)
    t = np.unique(matching[:, 1]).reshape(len(np.unique(matching[:, 1])), 1)
    t_tot = np.full_like(ll2, t)
    t_plot = uu.rescale_time(t_tot) if rescale else t_tot

    # Call the generic plotting function
    plot_generic(
        x=t_plot.T,   # Provide time values for each line (either one for each model or just one for single prediction)
        y=ll2.T,       # Multiple L2 error lines to plot
        title="Prediction error norm",
        xlabel=xlabel,
        ylabel=r"$L^2$ norm",
        legend_labels=legend_labels,  # Labels for the legend
        size=(6, 5),
        filename=f"{folder}/l2_{f'mm_{n_obs}obs' if MultiObs else f'obs{number}'}.png",
        colors=colors,
        linestyles=linestyles
    )


def plot_l2_matlab(X, theta_true, y_obs, y_mm_obs, folder):
    theta_true = theta_true.reshape(len(X), 1)
    t = np.unique(X[:, 1:2])
    # t_filtered = t[t > 0.000] 
    conf = OmegaConf.load(f"{folder}/config.yaml")
    n_obs = conf.model_parameters.n_obs
    obs_colors = uu.get_obs_colors(conf)
    obs_linestyles = uu.get_obs_linestyles(conf)
    _, mm_obs_color = uu.get_sys_mm_colors(conf)
    _, mm_obs_linestyle = uu.get_sys_mm_linestyle(conf)

    l2 = uu.calculate_l2(X, theta_true, y_mm_obs)
    l2_individual = []
    for i in range(n_obs):
        l2_individual.append(uu.calculate_l2(X, theta_true, y_obs[:, i]))

    ee = np.array(l2_individual).T
    l2 = l2.reshape(len(l2), 1)
    ll2 = np.hstack((l2, ee))

    # Generate corresponding legend labels
    legend_labels = ['MultiObs'] + [f'Obs {i}' for i in range(n_obs)]
    colors = [mm_obs_color] + obs_colors 
    linestyles = [mm_obs_linestyle] + obs_linestyles

    rescale = conf.plot.rescale
    _, xlabel, _ = uu.get_scaled_labels(rescale)

    t = t.reshape(len(t), 1)
    t_tot = np.full_like(ll2, t)
    t_plot = uu.rescale_time(t_tot) if rescale else t_tot


    # Call the generic plotting function
    plot_generic(
        x=t_plot.T,   # Provide time values for each line (either one for each model or just one for single prediction)
        y=ll2.T,       # Multiple L2 error lines to plot
        title="Prediction error norm",
        xlabel=xlabel,
        ylabel=r"$L^2$ norm",
        legend_labels=legend_labels,  # Labels for the legend
        size=(6, 5),
        filename=f"{folder}/l2_matlab_{n_obs}obs.png",
        colors=colors,
        linestyles=linestyles
    )



def plot_tf(tot_true, tot_obs_pred, number, prj_figs, MultiObs=False):
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
    match = uu.extract_matching(tot_true, tot_obs_pred)

    true = match[match[:, 1] == match[:, 1].max()][:, 2]
    true = true.reshape(len(true), 1)

    x_true = np.unique(tot_true[:, 0])
    x_pred = np.unique(tot_obs_pred[:, 0])

    conf = OmegaConf.load(f'{prj_figs}/config.yaml')
    n_obs = conf.model_parameters.n_obs
    obs_colors = uu.get_obs_colors(conf)
    true_color, mm_obs_color = uu.get_sys_mm_colors(conf)
    obs_linestyles = uu.get_obs_linestyles(conf)
    true_linestyle, mm_obs_linestyle = uu.get_sys_mm_linestyle(conf)

    if MultiObs:
        multi_pred = tot_obs_pred[:, -1][-len(x_pred):].reshape(len(x_pred), 1)

        # Generate individual predictions from each model in the ensemble
        individual_preds = [tot_obs_pred[:, 3+m][-len(x_pred):].reshape(len(x_pred), 1) for m in range(n_obs)]  # Assuming model.models holds individual models
        
        # Stack all predictions (true values + individual predictions + combined prediction)
        all_preds = [true] + individual_preds + [multi_pred]
        
        # x values: use xtr for true data, and 'x' for predictions
        x_vals = [x_true] + [x_pred for _ in range(n_obs + 1)]

        # Generate corresponding legend labels
        legend_labels = ['True'] + [f'Obs {i}' for i in range(len(individual_preds))] + ['MultiObs Pred']

        colors = [true_color] + obs_colors + [mm_obs_color]
        linestyles = [true_linestyle] + obs_linestyles + [mm_obs_linestyle]
    else:
        pred = tot_obs_pred[:, 3+number][-len(x_pred):].reshape(len(x_pred), 1)
        
        # Only two lines to plot: true and single prediction
        all_preds = [true, pred]
        x_vals = [x_true, x_pred]  # xtr for true, x for predicted
        legend_labels = ['True', f'Obs {number}']
        colors = [true_color] + [obs_colors[number]]
        linestyles = [true_linestyle] + [obs_linestyles[number]]

    rescale = conf.plot.rescale
    xlabel, _, ylabel = uu.get_scaled_labels(rescale)

    # x_vals = np.array(x_vals, dtype=float)
    x_vals_plot = uu.rescale_x(x_vals) if rescale else x_vals
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
        filename=f"{prj_figs}/tf_{f'mm_{n_obs}obs' if MultiObs else f'obs{number}'}.png",
        colors = colors,
        linestyles=linestyles
    )


# def plot_tf_matlab(e, theta_true, theta_observers, theta_mmobs, conf, prj_figs):
def plot_tf_matlab(e, theta_true, theta_observers, theta_mmobs, prj_figs):
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
    conf = OmegaConf.load(f"{src_dir}/config.yaml")
    number = conf.model_parameters.n_obs

    # Generate corresponding legend labels
    legend_labels = [f'Obs {i}' for i in range(number)] + ['True'] + ['MultiObs']

    obs_colors = uu.get_obs_colors(conf)
    system_color, mm_obs_color = uu.get_sys_mm_colors(conf)
    obs_linestyles = uu.get_obs_linestyles(conf)
    system_linestyle, mm_obs_linestyle = uu.get_sys_mm_linestyle(conf)

    colors = obs_colors + [system_color] + [mm_obs_color]
    linestyles = obs_linestyles + [system_linestyle] + [mm_obs_linestyle]
    rescale = conf.plot.rescale
    xlabel, _, ylabel = uu.get_scaled_labels(rescale)

    x_plot = uu.rescale_x(x_vals) if rescale else x_vals
    y_plot = uu.rescale_t(all_preds) if rescale else all_preds

    x_plot = np.array(x_plot)
    y_plot = np.array(y_plot)

    # Call the generic plotting function
    plot_generic(
        x=x_plot.T,  # Different x values for true and predicted
        y=y_plot.T,  # List of true and predicted lines
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

    conf = OmegaConf.load(f"{run_figs}/config.yaml")
    rescale = conf.plot.rescale
    n = conf.model_parameters.n_obs

    fname = f"{run_figs}/comparison_3d_matlab_{n}obs.png" if gt else f"{run_figs}/comparison_3d_pinns_{n}obs.png"

    theta_true_plot = uu.rescale_t(theta_true) if rescale else theta_true
    theta_pred_plot = uu.rescale_t(theta_pred) if rescale else theta_pred
    e_plot = np.hstack((uu.rescale_x(e[:, 0:1]), uu.rescale_time(e[:, 1:2]))) if rescale else e
    
    xlabel, ylabel, zlabel = uu.get_scaled_labels(rescale)
    # Call plot_generic_3d with the data
    plot_generic_3d(
        e_plot,                       # 2D array containing X and Y coordinates
        theta_true_plot,              # Surface 1: true values (Z1)
        theta_pred_plot,              # Surface 2: predicted values (Z2)
        xlabel, ylabel, zlabel,
        col_titles,      # Titles for each subplot
        fname
    )


def plot_timeseries_with_predictions(df, y1_pred, gt1_pred, gt2_pred, y2_pred, prj_figs, gt=False):
    conf = OmegaConf.load(f"{src_dir}/config.yaml")

    time_in_minutes = df['tau']*conf.model_properties.tauf / 60
    time_matlab = np.linspace(0, time_in_minutes.max(), len(y1_pred))
    
    # Prepare y-axis data (ground truth and predicted values)
    y_data = [
        df['y1'], df['gt1'], df['gt2'], df['y2'],  # Ground truth lines
        y1_pred, gt1_pred, gt2_pred, y2_pred       # Predicted lines
    ]

    # Labels for the legend (corresponding to each line in y_data)
    if gt:
        legend_labels = ['y1 (True)', 'gt1 (True)', 'gt2 (True)', 'y2 (True)', 
                        'y1 (Matlab)', 'gt1 (Matlab)', 'gt2 (Matlab)', 'y2 (Matlab)']
    else:
        legend_labels = ['y1 (True)', 'gt1 (True)', 'gt2 (True)', 'y2 (True)', 
                        'y1 (Pred)', 'gt1 (Pred)', 'gt2 (Pred)', 'y2 (Pred)']

    colors_points = conf.plot.colors.measuring_points
    colors_list = list(colors_points)
    colors = colors_list[:-1] * 2
    linestyles=["-", "-", "-", "-", "--", "--", "--", "--"]
    rescale = conf.plot.rescale
    _, _, ylabel = uu.get_scaled_labels(rescale)

    times = [time_in_minutes]*4 + [time_matlab]*4 

    if rescale:
        y_data_plot = uu.rescale_t(y_data)

    else:
        y_data_plot = y_data
    
    exp_type = conf.experiment.name
    n_obs = conf.model_parameters.n_obs
    type_dict = getattr(conf.experiment.type, exp_type[0])
    meas_dict = getattr(type_dict, exp_type[1])
    name = meas_dict["title"]
    # Call the generic plotting function
    plot_generic(
        x=times,        # Time data
        y=y_data_plot,       # All y data (ground truth + predictions)
        title=name,
        xlabel="Time (min)",
        ylabel=ylabel,
        legend_labels=legend_labels,
        colors=colors,
        linestyles=linestyles,
        size=(12, 6),
        filename=f"{prj_figs}/timeseries_vs_matlab_{n_obs}obs.png"
    )


def plot_tf_matlab_1obs(e, theta_true, theta_observer, prj_figs):
    
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
    tot = np.hstack((e, theta_true, theta_observer))
    final = tot[tot[:, 1] == 1]  # Select rows where time equals 1
    xtr = np.unique(tot[:, 0])   # Depth values for true data
    true = final[:, 2]          # True values at final time
    observer = final[:, 3:4]

    true_reshaped = true.reshape(len(xtr), 1)
    all_preds = np.hstack((observer, true_reshaped))
    # x values: use xtr for true data, and 'x' for predictions
    xxtr = xtr.reshape(len(xtr), 1)
    x_vals = np.full_like(all_preds, xxtr)
    conf = OmegaConf.load(f"{src_dir}/config.yaml")
    number = conf.model_parameters.n_obs

    # Generate corresponding legend labels
    legend_labels = [f'Obs {i}' for i in range(number)] + ['True']

    obs_colors = uu.get_obs_colors(conf)
    system_color, _ = uu.get_sys_mm_colors(conf)
    obs_linestyles = uu.get_obs_linestyles(conf)
    system_linestyle, _ = uu.get_sys_mm_linestyle(conf)

    colors = obs_colors + [system_color]
    linestyles = obs_linestyles + [system_linestyle]
    rescale = conf.plot.rescale
    xlabel, _, ylabel = uu.get_scaled_labels(rescale)

    x_plot = uu.rescale_x(x_vals) if rescale else x_vals
    y_plot = uu.rescale_t(all_preds) if rescale else all_preds


    # Call the generic plotting function
    plot_generic(
        x=x_plot.T,  # Different x values for true and predicted
        y=y_plot.T,  # List of true and predicted lines
        title="Prediction at final time",
        xlabel=xlabel,
        ylabel=ylabel,
        legend_labels=legend_labels,  # Labels for the legend
        size=(6, 5),
        colors=colors,
        filename=f"{prj_figs}/tf_mmobs_matlab_{number}obs.png",
        linestyles=linestyles
    )


def plot_l2_matlab_1obs(X, theta_true, y_obs, folder):
    theta_true = theta_true.reshape(len(X), 1)
    t = np.unique(X[:, 1:2])
    # t_filtered = t[t > 0.000] 
    conf = OmegaConf.load(f"{folder}/config.yaml")
    n_obs = conf.model_parameters.n_obs
    obs_colors = uu.get_obs_colors(conf)
    obs_linestyles = uu.get_obs_linestyles(conf)

    l2 = uu.calculate_l2(X, theta_true, y_obs)

    l2 = l2.reshape(len(l2), 1)


    # Generate corresponding legend labels
    legend_labels = [f'Obs {i}' for i in range(n_obs)]
    colors = obs_colors 
    linestyles = obs_linestyles

    rescale = conf.plot.rescale
    _, xlabel, _ = uu.get_scaled_labels(rescale)

    t = t.reshape(len(t), 1)
    t_tot = np.full_like(l2, t)
    t_plot = uu.rescale_time(t_tot) if rescale else t_tot


    # Call the generic plotting function
    plot_generic(
        x=t_plot.T,   # Provide time values for each line (either one for each model or just one for single prediction)
        y=l2.T,       # Multiple L2 error lines to plot
        title="Prediction error norm",
        xlabel=xlabel,
        ylabel=r"$L^2$ norm",
        legend_labels=legend_labels,  # Labels for the legend
        size=(6, 5),
        filename=f"{folder}/l2_matlab_{n_obs}obs.png",
        colors=colors,
        linestyles=linestyles
    )

def plot_t0(tot_pred, conf, out_dir):
    rescale = conf.plot.rescale
    set = conf.experiment.name

    e = uu.import_testdata(f"{set[0]}_{set[1]}")
    measurements_ic = e[e[:, 1]==0][:,2]

    x_measurements_ic = uu.get_tc_positions()

    oo = np.hstack(uu.gen_testdata(conf))
    x_matlab_ic = np.unique(oo[:, 0])
    matlab_ic = oo[oo[:, 1]==0][:,-1]

    x_pinns = np.unique(tot_pred[:, 0])
    pinns_ic = tot_pred[:, -1][:len(x_pinns)]

    x = [x_pinns, x_matlab_ic, x_measurements_ic]
    y = [pinns_ic, matlab_ic, measurements_ic]

    title = "Comparison at t=0" if rescale else r"Comparison at $\tau=0$"
    xlabel, _, ylabel = uu.get_scaled_labels(rescale)
    legend_labels = ["Obs PINNs", "Obs MATLAB", "Measurements"]
    fname = f"{out_dir}/t0_{set[0]}_{set[1]}.png"
    sys_colors, mm_colors = uu.get_sys_mm_colors(conf)
    obs_colors = uu.get_obs_colors(conf)
    sys_linestyle, obs_linestyle = uu.get_sys_mm_linestyle(conf)
    colors = [obs_colors[1], mm_colors, sys_colors]
    linestyles = ["-", obs_linestyle, sys_linestyle]

    x_plot = uu.rescale_x(x) if rescale else x
    y_plot = uu.rescale_t(y) if rescale else y

    plot_generic(x=x_plot,
                    y=y_plot,
                    title=title,
                    xlabel = xlabel,
                    ylabel=ylabel,
                    legend_labels=legend_labels,
                    filename=fname,
                    colors=colors,
                    linestyles=linestyles)
    


def plot_mm_obs(multi_obs, tot_true, tot_pred, config, output_dir, comparison_3d=True):
    
    t = np.unique(tot_pred[:, 1:2])
    mus = uu.mu(multi_obs, t)
    

    # if run_wandb:
    #     print(f"Initializing wandb for multi observer ...")
    #     wandb.init(project= str, name=f"mm_obs")

    matching = uu.extract_matching(tot_true, tot_pred)
    metrics = uu.compute_metrics(matching[:, 2], matching[:, 3])
    
    # if run_wandb:
    #     wandb.log(metrics)
    #     wandb.finish()

    plot_mu(mus, t, output_dir)
    plot_l2(tot_true, tot_pred, 0, output_dir, MultiObs=True)
    plot_tf(tot_true, tot_pred, 0, output_dir, MultiObs=True)
    plot_t0(tot_pred, config, output_dir)
    if comparison_3d:
        matching = uu.extract_matching(tot_true, tot_pred)
        plot_comparison_3d(matching[:, 0:2], matching[:, 3], matching[:, 4], output_dir)