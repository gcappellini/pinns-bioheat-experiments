import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
import deepxde as dde
import torch
import utils as uu
from omegaconf import OmegaConf
import datetime
from hydra import compose
# import v1v2_calc as cc 

# Set up directories and random seed
# dde.config.set_random_seed(200)
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)

models_dir = os.path.join(git_dir, "models")
os.makedirs(models_dir, exist_ok=True)

def plot_generic(x, y, title, xlabel, ylabel, legend_labels=None, log_scale=False, log_xscale=False, 
                 size=(6, 5), filename=None, colors=None, linestyles=None, markers=None,
                 linewidths=None, markersizes=None, alphas=None, markevery=50):
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
        color = colors[i] if isinstance(colors, list) and colors else None  # Use provided colors or default
        linestyle = linestyles[i] if isinstance(linestyles, list) and linestyles else (linestyles if linestyles else '-')  # Default to solid line
        linewidth = linewidths[i] if isinstance(linewidths, list) and linewidths else (linewidths if linewidths else 1.2)
        marker = markers[i] if isinstance(markers, list) and markers else (markers if markers else None)
        markersize = markersizes[i] if markersizes else 12
        alpha=alphas[i] if isinstance(alphas, list) and alphas else (alphas if alphas else 1)

        ax.plot(xi, yi, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth, 
                markersize=markersize, alpha=alpha, markevery=markevery)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add legend if labels are provided
    if legend_labels:
        ax.legend()

    # Set y-axis to log scale if specified
    if log_scale:
        ax.set_yscale('log')
    
    if log_xscale:
        ax.set_xscale('log')

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
def plot_loss_components(losshistory, nam, fold):
    loss_train, loss_test, iters = losshistory["train"], losshistory["test"], losshistory["steps"]
    # Prepare the loss data
    train = loss_train.sum(axis=1).ravel()
    test = loss_test.sum(axis=1).ravel()

    # Extract individual loss components
    # loss_res, loss_bc0, loss_bc1, loss_ic = loss_train[:, 0], loss_train[:, 1], loss_train[:, 2], loss_train[:, 3]
    loss_res, loss_bc0 = loss_train[:, 0], loss_train[:, 1]
    # Combine all loss components into a list/2D array for plotting
    # loss_terms = np.vstack((loss_res, loss_bc0, loss_bc1, loss_ic, test, train))
    loss_terms = np.vstack((loss_res, loss_bc0, test, train))
    
    # Labels for the legend
    # legend_labels = [r'$\mathcal{L}_{res}$', r'$\mathcal{L}_{bc0}$', r'$\mathcal{L}_{bc1}$', r'$\mathcal{L}_{ic}$', 'test loss', 'train loss']
    loss_names = uu.get_loss_names()
    # Get iterations (x-axis)
    iterations = np.array([iters]*len(loss_terms))
    conf = compose(config_name='config_run')
    plot_params = uu.get_plot_params(conf)
    colors = [plot_params[label]["color"] for label in loss_names]
    legend_labels = [plot_params[lab]["label"] for lab in loss_names]
    # colors = conf.plot.colors.losses

    # data_filename = f"{models_dir}/{str(datetime.date.today())}_losses_{nam}.npz"
    # np.savez(data_filename, iterations=iters, loss_res=loss_res, loss_bc0=loss_bc0, test=test, train=train, runtime=runtime)
    # fold = models_dir if fold is None else fold

    # Call the generic plotting function
    plot_generic(
        x=iterations,
        y=loss_terms,
        title="Loss Components",
        xlabel="iterations",
        ylabel="loss",
        legend_labels=legend_labels,
        log_scale=True,  # We want a log scale on the y-axis
        filename=f"{fold}/{str(datetime.date.today())}_losses_{nam}.png",
        colors=colors
    )


def plot_weights(series_data, run_figs, lal):

    conf = compose(config_name='config_run')
    plot_params = uu.get_plot_params(conf)
    pars = conf.model_parameters
    lam = pars.lam
    ups = pars.upsilon

    times = np.unique(series_data[0]['grid'][:, 1])

    # Prepare the labels for each weight line
    legend_labels = []
    colors = []
    linestyles = []
    run_figs
    rescale = conf.plot.rescale
    alphas = []
    linewidths = []
    weights = []
    t_vals = []

    for series in series_data:
        values = series['weights']
        label = series['label']
        colors.append(plot_params[label]["color"])
        linestyles.append(plot_params[label]["linestyle"])
        alphas.append(plot_params[label]["alpha"])
        linewidths.append(plot_params[label]["linewidth"])
        t_vals.append(times)
        weights.append(values.reshape(len(values), 1))
        legend_labels.append(plot_params[label]["label"])

    # Define the title with the lambda value
    title = fr"Dynamic weights, $\lambda={lam}$, $\upsilon={ups}$"
    
    times_plot = uu.rescale_time(t_vals) if rescale else t_vals
    _, xlabel, _ = uu.get_scaled_labels(rescale)
    weights = np.array(weights)
    times_plot = np.array(times_plot) 
    
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
        filename=f"{run_figs}/weights_{lal}.png",  # Filename to save the plot
        alphas=alphas,
        linewidths=linewidths
    )


def plot_obs_err(series_data: list, run_figs: str, lal: str):

    conf = compose(config_name='config_run')
    plot_params = uu.get_plot_params(conf)
    # Prepare the labels for each line based on the number of columns in `mus`

    x_ref = uu.get_tc_positions()
    intern_positions_dict = {k: v for k, v in x_ref.items() if k not in [ "y1"]}
    obs_positions = list(intern_positions_dict.values())
    obs_positions = [round(el, 2) for el in obs_positions]

    for xref in obs_positions:

        colors = []
        linestyles = []
        rescale = conf.plot.rescale
        alphas = []
        linewidths = []
        mus = []
        t_vals = []
        legend_labels = []

        for series in series_data:
            values = series[f'obs_err_{xref}']
            label = series['label']
            t_vals.append(np.unique(series["grid"][:, 1]))
            colors.append(plot_params[label]["color"])
            linestyles.append(plot_params[label]["linestyle"])
            alphas.append(plot_params[label]["alpha"])
            linewidths.append(plot_params[label]["linewidth"])
            mus.append(values.reshape(len(values), 1))
            legend_labels.append(plot_params[label]["label"])

        # Define the title for the plot
        title = f"Observation errors, {round(uu.rescale_x(xref)*100,0)} cm depth" if rescale else f"Observation errors, X={xref}"
        # t = t.reshape(len(t), 1)
        mus = np.array(uu.rescale_t(mus))-conf.temps.Troom if rescale else np.array(mus)  

        times_plot = np.array(uu.rescale_time(t_vals)) if rescale else np.array(t_vals)  
        _, xlabel, _ = uu.get_scaled_labels(rescale) 

        # Call the generic plotting function
        plot_generic(
            x=times_plot,                       # Time data for the x-axis
            y=mus,                   # Transpose mus to get lines for each observation error
            title=title,               # Plot title
            xlabel=xlabel,      # x-axis label
            ylabel=r"Error $^{\circ} C$" if rescale else r"Error",
            legend_labels=legend_labels, # Labels for each observation error
            size=(6, 5),               # Figure size
            colors=colors,
            linestyles=linestyles,
            filename=f"{run_figs}/obs_error_{xref}_{lal}.png",  # Filename to save the plot
            alphas=alphas,
            linewidths=linewidths
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
    conf = compose(config_name='config_run')
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
    conf = compose(config_name='config_run')
    n = conf.model_parameters.n_obs
    rescale = conf.plot.rescale
    # Determine the unique points in X and Y dimensions
    la = len(np.unique(e[:, 0]))  # Number of unique X points
    le = len(np.unique(e[:, 1]))  # Number of unique Y points
    
    # Reshape the true and predicted theta values to match the 2D grid
    theta_true = t_true.reshape(le, la)
    theta_pred = t_pred.reshape(le, la)

    # Column titles for each subplot
    col_titles = ["System", "Observer", "Error"] if n==1 else ["System", "MultiObserver", "Error"]

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


def plot_validation_3d(e, t_true, t_pred, run_figs, label):
    """
    Refactor the plot_comparison function to use plot_generic_3d for 3D comparisons.
    
    :param e: 2D array for X and Y data.
    :param t_true: True values reshaped for 3D plotting.
    :param t_pred: Predicted values reshaped for 3D plotting.
    :param run_figs: Directory to save the plot.
    """
    conf = compose(config_name='config_run')

    # Determine the unique points in X and Y dimensions
    la = len(np.unique(e[:, 0]))  # Number of unique X points
    le = len(np.unique(e[:, 1]))  # Number of unique Y points
    
    # Reshape the true and predicted theta values to match the 2D grid
    theta_true = t_true.reshape(le, la)
    theta_pred = t_pred.reshape(le, la)

    # Column titles for each subplot
    if label == "ground_truth":
        col_titles = ["Matlab System", "Matlab MM-Observer", "Error"]
    elif label.startswith("simulation") or label.startswith("inverse"):
        col_titles = ["Matlab", "PINNs", "Error"]
    elif label.startswith("meas"):
        col_titles = ["Measurements", "PINNs", "Error"]
    

    rescale = conf.plot.rescale

    fname = f"{run_figs}/validation_3d_{label}.png"

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
    

def plot_timeseries_with_predictions(system_meas: dict, mm_obs: dict, conf, out_dir):

    label = conf.experiment.run
    gt = mm_obs["label"].endswith("gt")
    x_points = uu.get_tc_positions()
    tf = conf.properties.tf

    y_data = []
    for entry in x_points.keys():
        closest_indices_pred = np.where(np.abs(mm_obs["grid"][:, 0] - x_points[entry]) == np.min(np.abs(mm_obs["grid"][:, 0] - x_points[entry])))
        dict_pred = {
            "tau": np.unique(system_meas["grid"][:, 1])*tf / 60,
            "theta": mm_obs["theta"][closest_indices_pred],
            "label": f"{entry} (Matlab)" if gt else f"{entry} (Pred)"
        }
        closest_indices_meas = np.where(np.abs(system_meas["grid"][:, 0] - x_points[entry]) == np.min(np.abs(system_meas["grid"][:, 0] - x_points[entry])))
        dict_meas = {
            "tau": np.unique(system_meas["grid"][:, 1])*tf / 60,
            "theta": system_meas["theta"][closest_indices_meas],
            "label": f"{entry} (Meas)"
        }
        y_data.append(dict_meas)
        y_data.append(dict_pred)


    # preds = uu.point_predictions([system_meas, mm_obs])
    # df = uu.load_from_pickle(f"{src_dir}/data/vessel/{label}.pkl")

    # time_in_minutes = df['tau']*conf.model_properties.tauf / 60
    # time_pred = preds[0]["tau"]*conf.model_properties.tauf / 60
    
    # Prepare y-axis data (ground truth and predicted values)
    # y_data = [
    #     *[df[pred['label']] for pred in preds],  # Extract values based on the 'label' key in preds
    #     *[pred['theta'] for pred in preds]     # Predicted lines
    # ]

    # Labels for the legend (corresponding to each line in y_data)
    plot_params = uu.get_plot_params(conf)
    # lines_labels = [pred["label"] for pred in preds]
    colors = [plot_params[ll]["color"] for ll in x_points.keys() for _ in range(2)]
    # meas_labels = [f"{ll} (Meas)" for ll in lines_labels]
    # gt_labels = [f"{ll} (Matlab)" for ll in lines_labels]
    # pinns_labels = [f"{ll} (Pred)" for ll in lines_labels]
    legend_labels = [p["label"] for p in y_data]
    
    # colors_points = conf.plot.colors.measuring_points
    # colors_list = list(colors_points)
    # colors = colors_list[:-2] * 2
    linestyles=(["-"] + ["--"])*len(x_points)

    rescale = conf.plot.rescale
    _, _, ylabel = uu.get_scaled_labels(rescale)

    times = [p["tau"] for p in y_data]

    y_data_plot = [uu.rescale_t(p["theta"]) for p in y_data] if rescale else [p["theta"] for p in y_data]

    conf_meas = OmegaConf.load(f"{src_dir}/configs/experiments.yaml")
    meas_dict = getattr(conf_meas, label)
    name = meas_dict["title"]
    # if gt:
    #     legend_labels = ['y1 (Meas)', 'gt2 (Meas)', 'y2 (Meas)', 
    #                     'y1 (Matlab)', 'gt2 (Matlab)', 'y2 (Matlab)']
    # else:
    #     legend_labels = ['y1 (Meas)', 'gt2 (Meas)', 'y2 (Meas)', 
    #                     'y1 (Pred)', 'gt2 (Pred)', 'y2 (Pred)']
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
        filename=f"{out_dir}/timeseries_vs_pinns_{label}_matlab.png" if gt else f"{out_dir}/timeseries_vs_pinns_{label}.png"
    )


def plot_multiple_series(series_data, prj_figs, lal):
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
    conf = compose(config_name='config_run')
    rescale = conf.plot.rescale
    plot_params = uu.get_plot_params(conf)
    
    lims = []  # Store the scale (range) for each subplot

    # Loop through each time instant and create individual subplots
    for i, tx in enumerate(t_vals):
        
        for series in series_data:
            grid = series['grid']
            values = series['theta']
            label = series['label']
            
            # Get closest match to current time instant in the grid
            closest_value = np.abs(grid[:, 1] - tx).min()
            x_vals = np.unique(grid[:, 0])
            values_tx = values[np.abs(grid[:, 1] - tx) == closest_value][:len(x_vals)]
            
            
            # Retrieve plot parameters based on the label
            color = plot_params[label]["color"]
            linestyle = plot_params[label]["linestyle"]
            alpha = plot_params[label]["alpha"]
            linewidth = plot_params[label]["linewidth"]
            marker = plot_params[label]["marker"]
         
            # Rescale values if required
            x_vals_plot = uu.rescale_x(x_vals) if rescale else x_vals
            values_plot = uu.rescale_t(values_tx) if rescale else values_tx
            
            # Plot the series on the current subplot
            axes[i].plot(x_vals_plot, values_plot, label=plot_params[label]["label"],
                         color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, marker=marker)
        
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
        lims.append(axes[i].get_ylim())

    # scales = []
    # # Apply adjusted limits with harmonized scales to each subplot
    # for i, (y_min, y_max) in enumerate(lims):
    #     scales.append(y_max-y_min)

    # max_scale = 6 if rescale else 0.5

    # for i, scale in enumerate(scales):
    #     if scale<=max_scale:
    #         scales[i]=max_scale
    #     axes[i].set_ylim(y_min, y_min + scales[i])
    
    axes[0].set_ylim(conf.temps.Troom, conf.temps.Tmax+0.7) if rescale else axes[0].set_ylim(0, 1)
    if conf.experiment.run.startswith("meas_cool"):
        axes[0].set_ylim(conf.temps.Troom, 30)
        
    # Save and close figure
    filename = f"{prj_figs}/combined_plot_{lal}.png"
    fig.tight_layout()  # Adjust layout for better spacing
    save_and_close(fig, filename)


def plot_l2(series_sys, series_data, folder, lal):
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

    conf = compose(config_name='config_run')
    plot_params = uu.get_plot_params(conf)

    t_vals = []
    ll2 = []
    legend_labels = []
    colors = []
    linestyles = []
    alphas = []
    linewidths = []

    for series in series_data:
        l2 = series['L2_err']
        label = series['label']

        # if label in ('theory', 'bound'):
        #     l2 = values
        # else:
        #     l2 = uu.calculate_l2(e, theta_system, values)

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
        ylabel=r"$L^2$ norm",
        legend_labels=legend_labels,  # Labels for the legend
        size=(6, 5),
        filename=f"{folder}/l2_combined_{lal}.png",
        colors=colors,
        linestyles=linestyles,
        alphas=alphas,
        linewidths=linewidths
    )


def plot_inv_var(cfg, var):
    iters = var["iters"]
    values = var["values"]
    true = np.full_like(values, cfg.parameters.wbsys)
    out_dir = cfg.output_dir
    plot_generic(
        x=[iters, iters] if cfg.run.startswith("simulation") else [iters],
        y=[values, true] if cfg.run.startswith("simulation") else [values],
        title=r"Recovered $w_b$ value",
        xlabel="Iterations",
        ylabel=r"$w_b \quad [s^{-1}]$",
        legend_labels=["PINNs", "MATLAB"] if cfg.run.startswith("simulation") else ["PINNs"],
        log_scale=True,
        log_xscale=False,
        size=(6, 5),
        filename=f"{out_dir}/variable_wb.png",
        colors=["cornflowerblue", "lightsteelblue"] if cfg.run.startswith("simulation") else ["cornflowerblue"],
        linestyles=["-", ":"] if cfg.run.startswith("simulation") else ["-"],
        markers=None,
        linewidths=None,
        markersizes=None,
        alphas=None,
        markevery=50
    )




def plot_res(config, system_gt=None, system=None, system_meas=None, observers_gt=None, observers=None, mm_obs=None, mm_obs_gt=None, var=None):
    hp, pars, plot = config.hp, config.parameters, config.plot
    out_dir = config.output_dir
    run = config.experiment.run
    weights_list = None
# blocco gt
    if plot.plot_gt:
        label = "ground_truth"
        multiple_series = [system_gt, mm_obs_gt]
        l2_ref_dict = system_gt
        ref_dict = mm_obs_gt
        validation_dict = mm_obs_gt
        l2_plot = [mm_obs_gt]
        timeseries_gt = system_gt
        timeseries_pred = mm_obs_gt
        if plot.show_obs:
            multiple_series.extend(observers_gt )
            l2_plot.extend(observers_gt)
            weights_list = observers_gt

        all_plots(multiple_series, out_dir, label, l2_ref_dict, l2_plot, ref_dict, validation_dict, config, timeseries_gt, timeseries_pred, weights_list)

# blocco direct
    if hp.nins==2:
        label = "direct"
        multiple_series = [system_gt, system]
        l2_ref_dict = system_gt
        ref_dict = l2_ref_dict
        validation_dict = system
        l2_plot = [system]
        timeseries_gt = system_gt
        timeseries_pred = system
# blocco inverse
        if var is not None:
            label= "inverse"
            plot_inv_var(config, var)
# blocco simulation_mm_obs    
    if hp.nins>2 and run.startswith("simulation"):
        label = "simulation_mm_obs"
        multiple_series = [system_gt, mm_obs]
        l2_ref_dict = system_gt
        validation_dict = mm_obs
        l2_plot = [mm_obs]
        timeseries_gt = system_gt
        timeseries_pred = mm_obs
        ref_dict = mm_obs_gt
# blocco measurement_mm_obs
    if hp.nins>2 and run.startswith("meas"):
        label = run
        multiple_series = [system_meas, mm_obs]
        l2_ref_dict = system_meas
        validation_dict = mm_obs
        l2_plot = [mm_obs]
        timeseries_gt = system_meas
        timeseries_pred = mm_obs
        ref_dict = mm_obs_gt
# blocco show_obs
    if plot.show_obs:
        multiple_series.extend(observers)
        l2_plot.extend(observers)
        weights_list = observers
    if plot.show_gt:
        multiple_series.append(mm_obs_gt)
        l2_plot.append(mm_obs_gt)

    all_plots(multiple_series, out_dir, label, l2_ref_dict, l2_plot, ref_dict, validation_dict, config, timeseries_gt, timeseries_pred, weights_list)


def all_plots(multiple_series, out_dir, label, l2_ref_dict, l2_plot, ref_dict, validation_dict, config, timeseries_gt, timeseries_pred, weights_list):
    nobs = config.parameters.nobs
    plot_multiple_series(multiple_series, out_dir, label)
    plot_l2(l2_ref_dict, l2_plot, out_dir, label)
    plot_validation_3d(ref_dict["grid"], ref_dict["theta"], validation_dict["theta"], out_dir, label)
    plot_obs_err(multiple_series[1:], out_dir, label)
    plot_timeseries_with_predictions(timeseries_gt, timeseries_pred, config, out_dir) 
    if 1 < nobs <= 8:
        plot_weights([*weights_list], out_dir, label)

# if __name__ == "__main__":