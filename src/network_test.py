import utils as uu
import os
import matlab.engine
import numpy as np
from scipy import integrate
import common as co
import plots as pp
import wandb
import argparse  # For handling command-line arguments

def setup_matlab_ground_truth(src_dir, run_matlab):
    """
    Optionally run the MATLAB script to compute ground truth.
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

def setup_wandb_logging(prj, run, config, use_wandb):
    """
    Optionally initialize wandb logging.
    """
    if use_wandb:
        print(f"Initializing wandb for project {prj} and run {run}...")
        wandb.init(project=prj, name=run, config=config)
    else:
        print("Skipping wandb logging.")

def finalize_wandb_logging(errors, use_wandb):
    """
    Optionally log errors to wandb and finish the run.
    """
    if use_wandb:
        wandb.log(errors)
        wandb.finish()

def main(run, run_matlab=False, use_wandb=False):
    """
    Main function to test the network. Ground truth via MATLAB and wandb logging are optional.
    """
    # Set up directories
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)

    # Optionally run MATLAB ground truth
    setup_matlab_ground_truth(src_dir, run_matlab)

    # Project and run setup
    prj = "network_test"
    co.set_prj(prj)
    run_figs = co.set_run(run)

    # Load configuration
    config = co.read_json(f"{src_dir}/properties.json")
    network_config = {
        "activation": config["activation"],
        "learning_rate": config["learning_rate"],
        "num_dense_layers": config["num_dense_layers"],
        "num_dense_nodes": config["num_dense_nodes"],
        "initialization": config["initialization"],
        "iterations": config["iterations"],
        "resampler_period": config["resampler_period"]
    }

    # Optionally set up wandb logging
    setup_wandb_logging(prj, run, network_config, use_wandb)
    co.write_json(config, f"{run_figs}/properties.json")

    # Plot Matlab
    t, weights = uu.load_weights()
    mu_matlab = uu.compute_mu()
    pp.plot_weights(weights, t, run_figs, gt=True)
    pp.plot_mu(mu_matlab, t, run_figs, gt=True)

    # Train the model
    model = uu.train_model(run_figs)

    # Generate test data
    X, y_sys, y_obs, _ = uu.gen_testdata()
    # t = np.unique(X[:, 1])
    x_obs = uu.gen_obsdata()
    
    # Model prediction
    y_pred = model.predict(x_obs)
    la = len(np.unique(X[:, 0]))
    le = len(np.unique(X[:, 1]))

    true = y_obs[:, 0].reshape(le, la)
    pred = y_pred.reshape(le, la)

    Xob_y2, y2_true = uu.gen_obs_y2()
    Xob_y1, y1_true = uu.gen_obs_y1()
    y2_pred, y1_pred = model.predict(Xob_y2), model.predict(Xob_y1)

    y = np.vstack([y2_true, y2_pred.reshape(y2_true.shape), y1_true, y1_pred.reshape(y2_true.shape)])
    legend_labels = [r'$y_2(\tau)$', r'$\hat{\theta}(0)$', r'$y_1(\tau)$', r'$\hat{\theta}(1)$']

    # Check Model prediction
    pp.plot_generic_3d(X[:, 0:2], pred, true, ["PINNs", "Matlab", "Error"], filename=f"{run_figs}/comparison_3d")
    pp.plot_generic(t, y, "Comparison at the boundary", r"Time ($\tau$)", r"Theta ($\theta$)", legend_labels, filename=f"{run_figs}/comparison_outputs")
    # Compute and log errors
    errors = uu.compute_metrics(y_obs[:, 0], y_pred)
    finalize_wandb_logging(errors, use_wandb)

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Network testing script.")
    parser.add_argument("--run_matlab", action="store_true", help="Run MATLAB ground truth.")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging.")
    args = parser.parse_args()

    run = "hpo_conf2_more_iterations"
    # Call main function with parsed arguments
    main(run, run_matlab=args.run_matlab, use_wandb=args.use_wandb)



    # python network_test.py --run_matlab --use_wandb