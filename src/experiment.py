import utils as uu
import os
import matlab.engine
import numpy as np
from scipy import integrate
import wandb
import plots as pp
import hydra
from omegaconf import DictConfig


# Function to run MATLAB ground truth if specified
def run_matlab_ground_truth(n_obs, output_dir, run_matlab, cfg):
    """
    Optionally run MATLAB ground truth.
    """
    if run_matlab:
        print("Running MATLAB ground truth calculation...")
        eng = matlab.engine.start_matlab()
        eng.cd(output_dir, nargout=0)
        eng.BioHeat(nargout=0)
        eng.quit()

        X, y_sys, _, y_mmobs = uu.gen_testdata(n_obs)
        t = np.unique(X[:, 1])

        mu = uu.compute_mu(n_obs)
        pp.plot_mu(mu, t, output_dir, gt=True)

        t, weights = uu.load_weights(n_obs)
        pp.plot_weights(weights, t, output_dir, cfg.model_parameters.lam, gt=True)
        pp.plot_comparison_3d(X, y_sys, y_mmobs, output_dir, gt=True)

        print("MATLAB ground truth completed.")
    else:
        print("Skipping MATLAB ground truth calculation.")


# Function to check observers and optionally upload results to wandb
def check_observers_and_wandb_upload(multi_obs, x_obs, X, y_sys, y_obs, run_wandb, output_dir, cfg):
    """
    Check observers and optionally upload results to wandb.
    """
    for el in range(len(multi_obs)):
        observer_dir = os.path.join(output_dir, f"observer_{el}")

        if run_wandb:
            # Use Hydra's runtime output directory name as the unique project name
            project_name = os.path.basename(cfg.output.base_dir)
            
            # Initialize wandb with config values from Hydra
            print(f"Initializing wandb for observer {el} with project name {project_name}...")
            wandb.init(
                project=project_name,          # Use the Hydra-generated output folder as the project name
                name=f"obs_{el}",
                config=cfg.model_properties    # Pass model properties from config
            )

        # Make predictions using the observer
        pred = multi_obs[el].predict(x_obs)
        y_sys = y_sys.reshape(pred.shape)

        # Plot the results
        pp.plot_l2(X, y_sys, pred, el, observer_dir)
        pp.plot_tf(X, y_sys, pred, multi_obs[el], el, observer_dir)

        # Test the observer and compute metrics
        metrics = uu.test_observer(multi_obs[el], observer_dir, X, x_obs, y_obs, el)

        # Optionally log to wandb
        if run_wandb:
            wandb.log(metrics)
            wandb.finish()


# Function to solve the IVP and plot weights, observer results
def solve_ivp_and_plot(multi_obs, n_obs, x_obs, X, y_sys, y_mm_obs, run_wandb, output_dir, cfg):
    """
    Solve the IVP for observer weights and plot the results.
    """
    p0 = np.full((n_obs,), 1/n_obs)
    lam = cfg.model_parameters.lam

    def f(t, p):
        a = uu.mu(multi_obs, t)
        e = np.exp(-1 * a)
        d = np.inner(p, e)
        f_list = []
        for el in range(len(p)):
            f_el = - lam * (1 - (e[:, el] / d)) * p[el]
            f_list.append(f_el)
        return np.array(f_list).reshape(len(f_list),)

    # Solve the IVP
    sol = integrate.solve_ivp(f, (0, 1), p0, t_eval=np.linspace(0, 1, 100))
    weights = np.zeros((sol.y.shape[0] + 1, sol.y.shape[1]))
    weights[0] = sol.t
    weights[1:] = sol.y

    # Save and plot weights
    np.save(f'{output_dir}/weights_lam_{lam}.npy', weights)
    pp.plot_weights(weights[1:], weights[0], output_dir)

    # Generate predictions
    y_pred = uu.mm_predict(multi_obs, lam, x_obs, output_dir)
    t = np.unique(X[:, 1:2])
    mus = uu.mu(multi_obs, t)

    if run_wandb:
        print(f"Initializing wandb for multi observer ...")
        wandb.init(project=cfg.project.name, name=f"mm_obs")

    # Compute and log metrics
    metrics = uu.compute_metrics(y_mm_obs, y_pred)

    if run_wandb:
        wandb.log(metrics)
        wandb.finish()

    # Plot results
    pp.plot_mu(mus, t, output_dir)
    pp.plot_l2(X, y_sys, y_pred, 0, output_dir, MultiObs=True)
    pp.plot_tf(X, y_sys, multi_obs, 0, output_dir, MultiObs=True)
    pp.plot_comparison_3d(X[:, 0:2], y_mm_obs, y_pred, f"{output_dir}/comparison_3d_mm_obs")
    pp.plot_observation_3d(X[:, 0:2], y_sys, y_pred, filename=f"{output_dir}/obs_3d_pinns")


# Main function with Hydra for configuration management
@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    # Access configuration values
    n_obs = cfg.model_parameters.n_obs
    output_dir = cfg.output.base_dir  # Hydra manages the output directory

    # Optionally run MATLAB ground truth
    run_matlab_ground_truth(n_obs, output_dir, cfg.experiment.run_matlab, cfg)

    # Uncomment below if you want to generate observers and solve IVP
    multi_obs = uu.mm_observer(n_obs)
    X, y_sys, y_obs, y_mm_obs = uu.gen_testdata(n_obs)
    x_obs = uu.gen_obsdata(n_obs)

    # Optionally check observers and upload to wandb
    check_observers_and_wandb_upload(multi_obs, x_obs, X, y_sys, y_obs, cfg.experiment.run_wandb, output_dir, cfg)

    # Optionally solve IVP and plot
    solve_ivp_and_plot(multi_obs, n_obs, x_obs, X, y_sys, y_mm_obs, cfg.experiment.run_wandb, output_dir, cfg)


if __name__ == "__main__":
    main()