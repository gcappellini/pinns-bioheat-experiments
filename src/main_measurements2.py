import utils as uu
import os
import matlab.engine
import numpy as np
from scipy import integrate
import common as co
import wandb
import plots as pp
import argparse
from import_vessel_data import load_measurements, extract_entries

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

a = co.read_json(f"{src_dir}/properties.json")
b = co.read_json(f"{src_dir}/parameters.json")

a["Ty20"] = b["y20_measured"]

co.write_json(a, f"{src_dir}/properties.json")


def check_observers_and_wandb_upload(multi_obs, x_obs, X, y_sys, run_wandb):
    """
    Check observers and optionally upload results to wandb.
    """
    for el in range(len(multi_obs)):
        run_figs = co.set_run(f"obs_{el}")
        aa = co.read_json(f"{run_figs}/properties.json")
        
        if run_wandb:
            print(f"Initializing wandb for observer {el}...")
            wandb.init(project="mm_obs_simulation", name=f"obs_{el}", config=aa)
        
        pred = multi_obs[el].predict(x_obs)
        
        pp.plot_l2(X, y_sys, pred, el, run_figs)
        pp.plot_tf(X, y_sys, multi_obs[el], el, run_figs)

        metrics = uu.compute_metrics(y_sys, pred)
 
        
        if run_wandb:
            wandb.log(metrics)
            wandb.finish()


def solve_ivp_and_plot(multi_obs, fold, n_obs, x_obs, X, y_sys, run_wandb):
    """
    Solve the IVP for observer weights and plot the results.
    """
    p0 = np.full((n_obs,), 1/n_obs)
    lam = b["lambda"]

    def f(t, p):
        a = uu.mu(multi_obs, t)
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
    pp.plot_weights(weights[1:], weights[0], fold)
    
    # Model prediction
    y_pred = uu.mm_predict(multi_obs, lam, x_obs, fold)
    t = np.unique(X[:, 1:2])
    mus = uu.mu(multi_obs, t)

    if run_wandb:
        print(f"Initializing wandb for multi observer ...")
        wandb.init(project=prj, name=f"mm_obs")

    metrics = uu.compute_metrics(y_sys, y_pred)
    
    if run_wandb:
        wandb.log(metrics)
        wandb.finish()

    pp.plot_mu(mus, t, fold)
    pp.plot_l2(X, y_sys, y_pred, 0, fold, MultiObs=True)
    pp.plot_tf(X, y_sys, multi_obs, 0, fold, MultiObs=True)
    pp.plot_observation_3d(X[:, 0:2], y_sys, y_pred, filename=f"{fold}/obs_3d_pinns")
    pp.plot_timeseries_with_predictions()


def scale_predictions(multi_obs, x_obs, prj_figs):
    """
    Generates and scales predictions from the multi-observer model.
    """
    positions = uu.get_tc_positions()
    lam = b["lambda"]
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
    
    return y1_pred, gt1_pred, gt2_pred, y2_pred


def main(n_obs, prj, run_wandb=False):
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """

    # Setup for PINNs implementation
    co.set_prj(prj)

    # Generate and check observers if needed
    multi_obs = uu.mm_observer(n_obs)
    a = uu.import_testdata()
    X = a[:, 0:2]
    meas = a[:, 2:3]
    x_obs = uu.import_obsdata()
    
    # Optionally check observers and upload to wandb
    check_observers_and_wandb_upload(multi_obs, x_obs, X, meas, run_wandb)

    run_figs = co.set_run(f"mm_obs")
    # Solve IVP and plot weights
    solve_ivp_and_plot(multi_obs, run_figs, n_obs, x_obs, X, meas, run_wandb)

    y1_pred, gt1_pred, gt2_pred, y2_pred = scale_predictions(multi_obs, x_obs, run_figs)

    # Load and plot timeseries data
    file_path = f"{src_dir}/data/vessel/20240522_1.txt"
    timeseries_data = load_measurements(file_path)
    df = extract_entries(timeseries_data, 83*60, 4*60*60)

    # Plot time series with predictions
    pp.plot_timeseries_with_predictions(df, y1_pred, gt1_pred, gt2_pred, y2_pred, run_figs)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run network testing with optional features.")
    parser.add_argument("--run_wandb", action="store_true", help="Use wandb for logging observers.")
    args = parser.parse_args()
    prj = "3Obs_meas2"
    n_obs = b["n_obs"]
    # Run main function with options
    main(n_obs, prj, run_wandb=args.run_wandb)