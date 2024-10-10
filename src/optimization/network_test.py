import utils as uu
import os
import matlab.engine
import numpy as np
from scipy import integrate
import common as co
import plots as pp
import wandb
import argparse  # For handling command-line arguments

def setup_matlab_ground_truth(src_dir, prj_figs, run_matlab):
    """
    Optionally run the MATLAB script to compute ground truth.
    """
    if run_matlab:
        print("Running MATLAB ground truth calculation...")
        eng = matlab.engine.start_matlab()
        eng.cd(src_dir, nargout=0)
        eng.BioHeat(nargout=0)
        eng.quit()

            # Plot Matlab
        t, weights = uu.load_weights()
        mu_matlab = uu.compute_mu()
        pp.plot_weights(weights, t, prj_figs, gt=True)
        pp.plot_mu(mu_matlab, t, prj_figs, gt=True)
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

def main(run, n_obs, run_matlab=False, use_wandb=False):
    """
    Main function to test the network. Ground truth via MATLAB and wandb logging are optional.
    """
    # Set up directories
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)


    # Project and run setup
    prj = "change_matlab1"
    prj_figs = co.set_prj(prj)
    run_figs = co.set_run(run)

    # Optionally run MATLAB ground truth
    setup_matlab_ground_truth(src_dir, prj_figs, run_matlab)

    # Load configuration
    config = co.read_json(f"{src_dir}/properties.json")
    param = co.read_json(f"{src_dir}/parameters.json")
    network_config = {
        "activation": config["activation"],
        "learning_rate": config["learning_rate"],
        "num_dense_layers": config["num_dense_layers"],
        "num_dense_nodes": config["num_dense_nodes"],
        "initialization": config["initialization"],
        "iterations": config["iterations"],
        "resampler_period": config["resampler_period"]
    }

    obs = f"W{n_obs}"
    config["W"]=param[obs]
    # Optionally set up wandb logging
    setup_wandb_logging(prj, run, network_config, use_wandb)
    co.write_json(config, f"{run_figs}/properties.json")
    co.write_json(config, f"{src_dir}/properties.json")


    # Train the model
    model = uu.train_model(run_figs)
    errors = uu.test_observer(model, run_figs, n_obs)

    finalize_wandb_logging(errors, use_wandb)

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Network testing script.")
    parser.add_argument("--run_matlab", action="store_true", help="Run MATLAB ground truth.")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging.")
    args = parser.parse_args()

    run = "obs_0"
    n_obs = 7
    # Call main function with parsed arguments
    main(run, n_obs, run_matlab=args.run_matlab, use_wandb=args.use_wandb)



    # python network_test.py --run_matlab --use_wandb