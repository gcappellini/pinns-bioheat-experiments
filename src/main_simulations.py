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

def run_matlab_ground_truth(src_dir, run_matlab):
    """
    Optionally run MATLAB ground truth.
    """
    if run_matlab:
        print("Running MATLAB ground truth calculation...")
        eng = matlab.engine.start_matlab()
        eng.cd(src_dir, nargout=0)
        eng.BioHeat(nargout=0)
        eng.quit()
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
            wandb.init(project="mm_obs_amc_cooling", name=f"new_obs_{el}", config=aa)
        
        pred = multi_obs[el].predict(x_obs)
        y_sys = y_sys.reshape(pred.shape)
        
        pp.check_obs(X, y_obs[:, el], pred, el, run_figs)
        pp.plot_l2_tf(X, y_sys, pred, multi_obs[el], el, run_figs)
        
        metrics = uu.compute_metrics(y_obs[:, el], pred)
        
        if run_wandb:
            wandb.log(metrics)
            wandb.finish()

def run_pinns(prj_figs):
    """
    Run the physics-informed neural networks (PINNs) implementation and generate plots.
    """
    X, y_sys, _, y_mmobs = uu.gen_testdata()
    t = np.unique(X[:, 1])

    mu = uu.compute_mu()
    pp.plot_mu(mu, t, prj_figs, gt=True)
    pp.plot_comparison(X, y_sys, y_mmobs, prj_figs)

    t, weights = uu.load_weights()
    pp.plot_weights(weights, t, prj_figs, gt=True)

def solve_ivp_and_plot(multi_obs, prj_figs, n_obs):
    """
    Solve the IVP for observer weights and plot the results.
    """
    x_obs = uu.gen_obsdata()
    p0 = np.full((n_obs,), 1/n_obs)
    par = co.read_json(f"{src_dir}/parameters.json")
    lam = par["lambda"]

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
    
    np.save(f'{prj_figs}/weights_lam_{lam}.npy', weights)
    pp.plot_weights(weights[1:], weights[0], prj_figs)

def main(run_matlab=False, run_wandb=False):
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """

    # Optionally run MATLAB ground truth
    run_matlab_ground_truth(src_dir, run_matlab)

    # Setup for PINNs implementation
    prj_figs = co.set_prj("test_matlab6")
    run_pinns(prj_figs)

    # Generate and check observers if needed
    multi_obs = uu.mm_observer()
    X, y_sys, y_obs, _ = uu.gen_testdata()
    x_obs = uu.gen_obsdata()
    
    # Optionally check observers and upload to wandb
    check_observers_and_wandb_upload(multi_obs, x_obs, X, y_sys, y_obs, run_wandb)

    # Solve IVP and plot weights
    solve_ivp_and_plot(multi_obs, prj_figs, n_obs=8)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run network testing with optional features.")
    parser.add_argument("--run_matlab", action="store_true", help="Run MATLAB ground truth.")
    parser.add_argument("--run_wandb", action="store_true", help="Use wandb for logging observers.")
    args = parser.parse_args()

    # Run main function with options
    main(run_matlab=args.run_matlab, run_wandb=args.run_wandb)