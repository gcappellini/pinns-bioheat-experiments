import utils as uu
import os
import matlab.engine
import numpy as np
from scipy import integrate
import common as co
import wandb
import plots as pp
import argparse
from omegaconf import OmegaConf

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)


def run_matlab_ground_truth(n_obs, src_dir, prj_figs, lam, run_matlab):
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
        pp.plot_weights(weights, t, prj_figs, lam, gt=True)
        pp.plot_comparison_3d(X, y_sys, y_mmobs, prj_figs, rescale= True, gt= True)

        print("MATLAB ground truth completed.")
    else:
        print("Skipping MATLAB ground truth calculation.")


def main(n_obs, prj_figs, conf, run_matlab=False, run_wandb=False):
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    lam = config.model_parameters.lam
    # Optionally run MATLAB ground truth
    run_matlab_ground_truth(n_obs, src_dir, prj_figs, lam, run_matlab)

    # # Generate and check observers if needed
    # multi_obs = uu.mm_observer(n_obs, conf)
    # X, y_sys, y_obs, y_mm_obs = uu.gen_testdata(n_obs)
    # x_obs = uu.gen_obsdata(n_obs)
    # uu.check_observers_and_wandb_upload(multi_obs, x_obs, X, y_sys, run_wandb, prj_figs)

    # run_figs = co.set_run(f"mm_obs")
    # config.model_properties.W = None
    # OmegaConf.save(config, f"{run_figs}/config.yaml") 
    # # Solve IVP and plot weights
    # uu.solve_ivp_and_plot(multi_obs, run_figs, n_obs, x_obs, X, y_sys, lam)


if __name__ == "__main__":
    # Parse the config path argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # Load the configuration from the passed file
    config = OmegaConf.load(args.config_path)

    # Now you can access your config values
    experiment_type = config.experiment.type
    print(f"Running measurement with experiment type: {experiment_type}")

    n_obs = config.model_parameters.n_obs
    prj_name = config.experiment.name
    
    output_dir = co.set_prj(prj_name)
    # Run main function with options
    OmegaConf.save(config,f"{output_dir}/config.yaml")
    main(n_obs, output_dir, config, run_matlab=config.experiment.run_matlab, run_wandb=config.experiment.run_wandb)