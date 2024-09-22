import utils as uu
import os
import matlab.engine
import numpy as np
from scipy import integrate
import common as co
import wandb
import plots as pp
import argparse

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

a = co.read_json(f"{src_dir}/properties.json")
b = co.read_json(f"{src_dir}/parameters.json")

a["Ty20"] = b["y20_simulated"]

co.write_json(a, f"{src_dir}/properties.json")

def run_matlab_ground_truth(n_obs, src_dir, prj_figs, run_matlab):
    """
    Optionally run MATLAB ground truth.
    """
    if run_matlab:
        print("Running MATLAB ground truth calculation...")
        eng = matlab.engine.start_matlab()
        eng.cd(src_dir, nargout=0)
        eng.BioHeat(nargout=0)
        eng.quit()

        X, y_sys, _, y_mmobs = uu.gen_testdata(n_obs)
        t = np.unique(X[:, 1])

        mu = uu.compute_mu(n_obs)
        pp.plot_mu(mu, t, prj_figs, gt=True)

        t, weights = uu.load_weights(n_obs)
        pp.plot_weights(weights, t, prj_figs, gt=True)

        print("MATLAB ground truth completed.")
    else:
        print("Skipping MATLAB ground truth calculation.")

def check_observers_and_wandb_upload(multi_obs, x_obs, X, y_sys, y_obs, run_wandb):
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
        y_sys = y_sys.reshape(pred.shape)
        
        pp.plot_l2(X, y_sys, pred, el, run_figs)
        pp.plot_tf(X, y_sys, pred, multi_obs[el], el, run_figs)

        metrics = uu.test_observer(multi_obs[el], run_figs, X, x_obs, y_obs, el)
 
        if run_wandb:
            wandb.log(metrics)
            wandb.finish()


def solve_ivp_and_plot(multi_obs, fold, n_obs, x_obs, X, y_sys, y_mm_obs, run_wandb):
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

    metrics = uu.compute_metrics(y_mm_obs, y_pred)
    
    if run_wandb:
        wandb.log(metrics)
        wandb.finish()

    pp.plot_mu(mus, t, fold)
    pp.plot_l2(X, y_sys, y_pred, 0, fold, MultiObs=True)
    pp.plot_tf(X, y_sys, multi_obs, 0, fold, MultiObs=True)
    pp.plot_comparison_3d(X[:, 0:2], y_mm_obs, y_pred, f"{fold}/comparison_3d_mm_obs")
    pp.plot_observation_3d(X[:, 0:2], y_sys, y_pred, filename=f"{fold}/obs_3d_pinns")


def main(n_obs, prj, run_matlab=False, run_wandb=False):
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """

    # Setup for PINNs implementation
    prj_figs = co.set_prj(prj)

    # Optionally run MATLAB ground truth
    run_matlab_ground_truth(n_obs, src_dir, prj_figs, run_matlab)

    # Generate and check observers if needed
    multi_obs = uu.mm_observer(n_obs)
    X, y_sys, y_obs, y_mm_obs = uu.gen_testdata(n_obs)
    x_obs = uu.gen_obsdata(n_obs)
    
    # Optionally check observers and upload to wandb
    check_observers_and_wandb_upload(multi_obs, x_obs, X, y_sys, y_obs, run_wandb)

    run_figs = co.set_run(f"mm_obs")
    # Solve IVP and plot weights
    solve_ivp_and_plot(multi_obs, run_figs, n_obs, x_obs, X, y_sys, y_mm_obs, run_wandb)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run network testing with optional features.")
    parser.add_argument("--run_matlab", action="store_true", help="Run MATLAB ground truth.")
    parser.add_argument("--run_wandb", action="store_true", help="Use wandb for logging observers.")
    args = parser.parse_args()
    prj = "3Obs"
    n_obs = b["n_obs"]
    # Run main function with options
    main(n_obs, prj, run_matlab=args.run_matlab, run_wandb=args.run_wandb)