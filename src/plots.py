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
        label = legend_labels[i] if isinstance(legend_labels, list) and legend_labels else (legend_labels if legend_labels else None)
        color = colors[i] if isinstance(colors, list) and colors else (colors if colors else None)  # Use provided colors or default
        linestyle = linestyles[i] if isinstance(linestyles, list) and linestyles else (linestyles if linestyles else '-')  # Default to solid line
        linewidth = linewidths[i] if isinstance(linewidths, list) and linewidths else (linewidths if linewidths else 1.2)
        marker = markers[i] if isinstance(markers, list) and markers else (markers if markers else None)
        markersize = markersizes[i] if markersizes else 12
        alpha=alphas[i] if isinstance(alphas, list) and alphas else (alphas if alphas else 1)

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
    ups = conf1.model_parameters.upsilon
    n_obs = conf1.model_parameters.n_obs

    # Prepare the labels for each weight line
    legend_labels = [f"Weight $p_{i}$" for i in range(weights.shape[0])]

    # Define the title with the lambda value
    title = fr"Dynamic weights, $\lambda={lam}$, $\upsilon={ups}$"
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
        filename=f"{run_figs}/weights_l_{lam}_u_{ups}_{'matlab' if gt else 'pinns'}_{n_obs}obs.png"  # Filename to save the plot
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


def plot_tx(tx, tot_true, tot_obs_pred, number, prj_figs, system=False, gt=False, MultiObs=False):
    """
    Plot true and predicted values (single or multi-observer model) for a given time step.
    
    :param tx: Time step to plot.
    :param tot_true: True data array (2D).
    :param tot_obs_pred: Predicted data array (2D).
    :param number: Observer identifier for plotting.
    :param prj_figs: Directory to save the plot figure.
    :param system: If True, plot system prediction.
    :param gt: If True, plot ground truth comparison.
    :param MultiObs: If True, plot multiple observer model predictions with a combined weighted average.
    """
    # Reshape inputs and match time steps
    match = uu.extract_matching(tot_true, tot_obs_pred)
    closest_value = np.abs(match[:, 1] - tx).min()
    true_tx = match[np.abs(match[:, 1] - tx) == closest_value][:, 2]
    preds_tx = tot_obs_pred[np.abs(tot_obs_pred[:, 1] - tx) == closest_value][:, 2:]
    true = true_tx.reshape(len(true_tx), 1)

    # x-axis values
    x_true = np.unique(tot_true[:, 0])
    x_pred = np.unique(tot_obs_pred[:, 0])

    # Load configuration and plot parameters
    conf = OmegaConf.load(f'{prj_figs}/config.yaml')
    plot_params = uu.get_plot_params(conf)
    n_obs = conf.model_parameters.n_obs

    # Generate prediction data for different scenarios
    if MultiObs:
        multi_pred = preds_tx[:, -1].reshape(len(x_pred), 1)
        individual_preds = [preds_tx[:, -(1 + n_obs) + m].reshape(len(x_pred), 1)
                            for m in range(n_obs)]
        all_preds = [true] + individual_preds + [multi_pred]
        x_vals = [x_true] + [x_pred for _ in range(n_obs + 1)]
        legend_labels = ['True'] + [f'Obs {i}' for i in range(len(individual_preds))] + ['MultiObs Pred']
    
    elif system:
        sys_pred = preds_tx[:, -1].reshape(len(x_pred), 1)
        all_preds = [true, sys_pred]
        x_vals = [x_true, x_pred]
        legend_labels = ['True', 'System Pred']

    else:
        pred = preds_tx[:, -1 if n_obs == 1 else 3 + number].reshape(len(x_pred), 1)
        all_preds = [true, pred]
        x_vals = [x_true, x_pred]
        legend_labels = ['True', f'Obs {number}']

    # Handle ground truth if applicable
    if gt:
        matlab_sol = uu.gen_testdata(conf)
        x_matlab = np.unique(matlab_sol[:, 0:1])
        if MultiObs:
            individual_sol = [matlab_sol[:, -(1 + n_obs) + m][-len(x_matlab):].reshape(len(x_matlab), 1)
                              for m in range(n_obs)]
            all_preds += individual_sol + [matlab_sol[:, -1][-len(x_matlab):].reshape(len(x_matlab), 1)]
            x_vals += [x_matlab for _ in range(n_obs + 1)]
            legend_labels += [f'MATLAB Obs {i}' for i in range(n_obs)] + ['MATLAB MultiObs']
        elif system:
            all_preds.append(matlab_sol[:, -1][-len(x_matlab):].reshape(len(x_matlab), 1))
            x_vals.append(x_matlab)
            legend_labels.append('MATLAB System Pred')
        else:
            matlab_obs = matlab_sol[:, 3 + number][-len(x_matlab):].reshape(len(x_matlab), 1)
            all_preds.append(matlab_obs)
            x_vals.append(x_matlab)
            legend_labels.append(f'MATLAB Obs {number}')

    # Rescale if needed
    rescale = conf.plot.rescale
    xlabel, _, ylabel = uu.get_scaled_labels(rescale)
    x_vals_plot = uu.rescale_x(x_vals) if rescale else x_vals
    all_preds_plot = uu.rescale_t(all_preds) if rescale else all_preds
    time = uu.rescale_time(tx)
    title = f"Prediction at t={time} s" if rescale else fr"Prediction at $\tau$={tx}"

    # Call the generic plotting function
    plot_generic(
        x=x_vals_plot,
        y=all_preds_plot,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend_labels=legend_labels,
        size=(6, 5),
        filename=f"{prj_figs}/t{tx}_{'multi_obs' if MultiObs else 'system' if system else f'obs{number}'}.png",
        **plot_params  # Pass colors, linestyles, markers, alphas, and linewidths
    )


def plot_comparison_3d(e, t_true, t_pred, run_figs, gt=False):
    """
    Refactor the plot_comparison function to use plot_generic_3d for 3D comparisons.
    
    :param e: 2D array for X and Y data.
    :param t_true: True values reshaped for 3D plotting.
    :param t_pred: Predicted values reshaped for 3D plotting.
    :param run_figs: Directory to save the plot.
    """
    conf = OmegaConf.load(f"{run_figs}/config.yaml")
    n_obs = conf.model_parameters.n_obs
    # Determine the unique points in X and Y dimensions
    la = len(np.unique(e[:, 0]))  # Number of unique X points
    le = len(np.unique(e[:, 1]))  # Number of unique Y points
    
    # Reshape the true and predicted theta values to match the 2D grid
    theta_true = t_true.reshape(le, la)
    theta_pred = t_pred.reshape(le, la)

    # Column titles for each subplot
    col_titles = ["System", "Observer", "Error"] if n_obs==1 else ["System", "MultiObserver", "Error"]

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


def plot_validation_3d(e, t_true, t_pred, run_figs, system=False):
    """
    Refactor the plot_comparison function to use plot_generic_3d for 3D comparisons.
    
    :param e: 2D array for X and Y data.
    :param t_true: True values reshaped for 3D plotting.
    :param t_pred: Predicted values reshaped for 3D plotting.
    :param run_figs: Directory to save the plot.
    """
    conf = OmegaConf.load(f"{run_figs}/config.yaml")

    # Determine the unique points in X and Y dimensions
    la = len(np.unique(e[:, 0]))  # Number of unique X points
    le = len(np.unique(e[:, 1]))  # Number of unique Y points
    
    # Reshape the true and predicted theta values to match the 2D grid
    theta_true = t_true.reshape(le, la)
    theta_pred = t_pred.reshape(le, la)

    # Column titles for each subplot
    col_titles = ["MATLAB", "PINNs", "Error"] 

    conf = OmegaConf.load(f"{run_figs}/config.yaml")
    rescale = conf.plot.rescale
    n = conf.model_parameters.n_obs

    fname = f"{run_figs}/validation_3d_system.png" if system else f"{run_figs}/validation_3d_{n}obs.png"

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
        filename=f"{prj_figs}/timeseries_vs_pinns_{n_obs}obs.png"
    )


def plot_mm_obs(multi_obs, tot_true, tot_pred, output_dir, comparison_3d=True):
    
    t = np.unique(tot_pred[:, 1:2])
    mus = uu.mu(multi_obs, t)
    

    # if run_wandb:
    #     print(f"Initializing wandb for multi observer ...")
    #     wandb.init(project= str, name=f"mm_obs")

    matching = uu.extract_matching(tot_true, tot_pred)
    uu.compute_metrics(matching[:, 2], matching[:, 3], output_dir)
    
    # if run_wandb:
    #     wandb.log(metrics)
    #     wandb.finish()

    plot_mu(mus, t, output_dir)
    plot_l2(tot_true, tot_pred, 0, output_dir, MultiObs=True)
    plot_generic_5_figs(tot_true, tot_pred, 0, output_dir, MultiObs=True)

    if comparison_3d:
        matching = uu.extract_matching(tot_true, tot_pred)
        plot_comparison_3d(matching[:, 0:2], matching[:, 2], matching[:, -1], output_dir)


def plot_multiple_series(series_data, prj_figs):
    """
    Generalized plot function for multiple series at specified time instants.
    
    :param series_data: List of dictionaries, each with keys: 'grid', 'theta', 'label'.
                        Each dictionary contains:
                          - 'grid': Array with shape (N, 2) containing x and t values.
                          - 'theta': Array of predicted/true values corresponding to 'grid'.
                          - 'label': String indicating the type (e.g., "system", "observer", "multi_obs", etc.).
    :param prj_figs: Directory to save the figure.
    """
    t_vals = [0, 0.25, 0.51, 0.75, 1]
    fig, axes = plt.subplots(1, len(t_vals), figsize=(15, 5))
    
    # Load configuration parameters for plotting
    cfg = OmegaConf.load(f'{prj_figs}/config.yaml')
    conf = OmegaConf.load(f"{cfg.output_dir}/config.yaml")
    rescale = conf.plot.rescale
    plot_params = uu.get_plot_params(conf)
    
    all_y_values = []  # Collect all y-values across all subplots for global scaling
    y_limits_per_subplot = []  # Store individual y-limits for each subplot

    # Loop through each time instant and create individual subplots
    for i, tx in enumerate(t_vals):
        subplot_y_values = []  # Track y-values for the current subplot
        
        for series in series_data:
            grid = series['grid']
            values = series['theta']
            label = series['label']
            
            # Get closest match to current time instant in the grid
            closest_value = np.abs(grid[:, 1] - tx).min()
            values_tx = values[np.abs(grid[:, 1] - tx) == closest_value]
            x_vals = np.unique(grid[:, 0])
            
            # Retrieve plot parameters based on the label
            color = plot_params[label]["color"]
            linestyle = plot_params[label]["linestyle"]
            alpha = plot_params[label]["alpha"]
            linewidth = plot_params[label]["linewidth"]
         
            # Rescale values if required
            x_vals_plot = uu.rescale_x(x_vals) if rescale else x_vals
            values_plot = uu.rescale_t(values_tx) if rescale else values_tx
            
            # Collect y-values for scaling
            subplot_y_values.extend(values_plot)
            all_y_values.extend(values_plot)
            
            # Plot the series on the current subplot
            axes[i].plot(x_vals_plot, values_plot, label=plot_params[label]["label"],
                         color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        
        # Compute y-limits for this subplot
        subplot_y_min, subplot_y_max = min(subplot_y_values), max(subplot_y_values)
        y_limits_per_subplot.append((subplot_y_min, subplot_y_max))
        
        # Set time, labels, and title for each subplot
        time = uu.rescale_time(tx) if rescale else tx
        title = f"Time t={time} s" if rescale else fr"Time $\tau$={tx}"
        xlabel, _, ylabel = uu.get_scaled_labels(rescale)
        
        # Configure subplot labels and title
        axes[i].set_xlabel(xlabel)
        if i == 0:
            axes[i].set_ylabel(ylabel)
            axes[i].legend(loc='best')
        axes[i].set_title(title, fontweight='bold')
        axes[i].grid(True)
    
    # Determine global margins based on all data
    global_y_min, global_y_max = min(all_y_values), max(all_y_values)
    global_margin = 0.1 * (global_y_max - global_y_min)  # 10% margin
    
    # Apply adjusted limits with margin to each subplot
    for i, (subplot_y_min, subplot_y_max) in enumerate(y_limits_per_subplot):
        # Expand subplot's limits to include a margin based on global extrema
        margin = 1.5 * (subplot_y_max - subplot_y_min)  # Local margin for visibility
        adjusted_y_min = max(global_y_min, subplot_y_min - margin)
        adjusted_y_max = min(global_y_max, subplot_y_max + margin)
        axes[i].set_ylim(adjusted_y_min, adjusted_y_max)
    
    # Save and close figure
    filename = f"{prj_figs}/combined_plot.png"
    fig.tight_layout()  # Adjust layout for better spacing
    save_and_close(fig, filename)


def plot_l2(series_sys, series_data, folder):
    """
    Plot L2 norm of prediction errors for true and predicted values.
    
    :param xobs: Input observations (depth and time).
    :param theta_true: True theta values.
    :param model: A single model or a list of models if MultiObs is True.
    :param number: Identifier for the observation.
    :param folder: Directory to save the figure.
    :param MultiObs: If True, use multiple models for predictions.
    """

    e = series_sys['grid']
    theta_system = series_sys['theta'].reshape(len(e), 1)
    t_pred = np.unique(e[:, 1])
    t_pred = t_pred.reshape(len(t_pred), 1)

    cfg = OmegaConf.load(f'{folder}/config.yaml')
    conf = OmegaConf.load(f"{cfg.output_dir}/config.yaml")
    plot_params = uu.get_plot_params(conf)

    t_vals = []
    ll2 = []
    legend_labels = []
    colors = []
    linestyles = []
    alphas = []
    linewidths = []

    for series in series_data:
        values = series['theta']
        label = series['label']

        if label in ('theory', 'bound'):
            l2 = values
        else:
            l2 = uu.calculate_l2(e, theta_system, values)
        l2 = l2.reshape(len(l2), 1)

        # t_vals = [t_pred for _ in range(n_obs + 1)]
        t_vals.append(t_pred)
        legend_labels.append(plot_params[label]["label"])
        colors.append(plot_params[label]["color"])
        linestyles.append(plot_params[label]["linestyle"])
        alphas.append(plot_params[label]["alpha"])
        linewidths.append(plot_params[label]["linewidth"])
        ll2.append(l2)


    # rescale = conf.plot.rescale if rescale==None else rescale
    rescale=False
    _, xlabel, _ = uu.get_scaled_labels(rescale)
    t_vals_plot = np.array(uu.rescale_time(t_vals)) if rescale else np.array(t_vals)
    ll2 = np.array(ll2)
    ll2 = ll2.reshape(len(series_data), len(t_pred))
    t_vals_plot = t_vals_plot.reshape(len(series_data), len(t_pred))

    # Call the generic plotting function
    plot_generic(
        x=t_vals_plot,   # Provide time values for each line (either one for each model or just one for single prediction)
        y=ll2,       # Multiple L2 error lines to plot
        title="Prediction error norm",
        xlabel=xlabel,
        ylabel=r"$L2$ norm",
        legend_labels=legend_labels,  # Labels for the legend
        size=(6, 5),
        filename=f"{folder}/l2_combined.png",
        colors=colors,
        linestyles=linestyles,
        alphas=alphas,
        linewidths=linewidths
    )