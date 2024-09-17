import utils as uu
import os
import common as co
import numpy as np
import wandb
import matplotlib.pyplot as plt
from scipy import integrate
from import_vessel_data import load_measurements, extract_entries
import plots as pp

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

a = co.read_json(f"{src_dir}/properties.json")
b = co.read_json(f"{src_dir}/parameters.json")

a["Ty20"] = b["y20_measured"]

co.write_json(a, f"{src_dir}/properties.json")

def initialize_project(project_name):
    """
    Initializes the project directories and setups for saving figures.
    """
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    prj_figs = co.set_prj(project_name)
    return src_dir, prj_figs

def read_parameters(src_dir):
    """
    Reads the parameters from the properties.json and parameters.json files.
    """
    properties_file = f"{src_dir}/properties.json"
    parameters_file = f"{src_dir}/parameters.json"
    
    properties = co.read_json(properties_file)
    parameters = co.read_json(parameters_file)
    return properties, parameters


def run_multi_observer_predictions(multi_obs, x_obs, X, meas, prj_figs):
    """
    Runs multi-observer predictions and plots the results.
    """
    for el in range(len(multi_obs)):
        run_figs = co.set_run(f"obs_{el}")
        pred = multi_obs[el].predict(x_obs)
        meas = meas.reshape(pred.shape)
        
        # Plot results using plot_l2_tf (no wandb integration here)
        pp.plot_l2_tf(X, meas, pred, multi_obs[el], el, run_figs)

        # Compute and log the metrics locally (not using wandb)
        metrics = uu.compute_metrics(meas, pred)
        # Optionally: Save metrics somewhere or print them
        print(f"Metrics for observer {el}: {metrics}")


def solve_ivp_for_weights(multi_obs, n_obs, lam, prj_figs):
    """
    Solves the IVP for observer weights using the provided multi-observer data.
    """
    p0 = np.full((n_obs,), 1 / n_obs)
    
    def ivp_function(t, p):
        a = uu.mu(multi_obs, t)
        e = np.exp(-1 * a)
        d = np.inner(p, e)
        f = []
        for el in range(len(p)):
            f_el = - lam * (1 - (e[:, el] / d)) * p[el]
            f.append(f_el)
        return np.array(f).reshape(len(f),)

    sol = integrate.solve_ivp(ivp_function, (0, 1), p0, t_eval=np.linspace(0, 1, 100))

    # Save weights and plot them
    weights = np.zeros((sol.y.shape[0] + 1, sol.y.shape[1]))
    weights[0] = sol.t
    weights[1:] = sol.y
    np.save(f'{prj_figs}/weights_lam_{lam}.npy', weights)
    pp.plot_weights(weights[1:], weights[0], prj_figs)
    
    return sol

def scale_and_plot_predictions(X, meas, multi_obs, x_obs, positions, prj_figs, lam):
    """
    Generates and scales predictions from the multi-observer model.
    """
    mm_obs_pred = uu.mm_predict(multi_obs, lam, x_obs, prj_figs)
    preds = np.vstack((x_obs[:, 0], x_obs[:, -1], mm_obs_pred)).T
    
    # Extract predictions based on positions
    y2_pred_sc = preds[preds[:, 0] == positions[0]][:, 2]
    gt2_pred_sc = preds[preds[:, 0] == positions[1]][:, 2]
    gt1_pred_sc = preds[preds[:, 0] == positions[2]][:, 2]
    y1_pred_sc = preds[preds[:, 0] == positions[3]][:, 2]
    
    # Rescale predictions
    y2_pred = uu.rescale_t(y2_pred_sc)
    gt2_pred = uu.rescale_t(gt2_pred_sc)
    gt1_pred = uu.rescale_t(gt1_pred_sc)
    y1_pred = uu.rescale_t(y1_pred_sc)

    # Plot L2 comparison (with MultiObs flag)
    pp.plot_l2_tf(X, meas, preds[:, -1], multi_obs, 0, prj_figs, MultiObs=True)
    
    return y1_pred, gt1_pred, gt2_pred, y2_pred


def main():
    # Initialization
    src_dir, prj_figs = initialize_project("test_cooling2")
    
    # Read properties and parameters
    properties, parameters = read_parameters(src_dir)
    lam = parameters["lambda"]

    # Generate multi-observer predictions
    multi_obs = uu.mm_observer()
    a = uu.import_testdata()
    X = a[:, 0:2]
    meas = a[:, 2:3]
    bolus = a[:, 3:4]
    x_obs = uu.import_obsdata()

    run_multi_observer_predictions(multi_obs, x_obs, X, meas, prj_figs)

    # Solve IVP for observer weights
    sol = solve_ivp_for_weights(multi_obs, n_obs=8, lam=lam, prj_figs=prj_figs)

    # Final predictions and scaling
    positions = uu.get_tc_positions()
    y1_pred, gt1_pred, gt2_pred, y2_pred = scale_and_plot_predictions(X, meas, multi_obs, x_obs, positions, prj_figs, lam)

    # Load and plot timeseries data
    file_path = f"{src_dir}/data/vessel/20240522_1.txt"
    timeseries_data = load_measurements(file_path)
    df = extract_entries(timeseries_data, 83*60, 4*60*60)

    # Plot time series with predictions
    pp.plot_timeseries_with_predictions(df, y1_pred, gt1_pred, gt2_pred, y2_pred, prj_figs)

if __name__ == "__main__":
    main()