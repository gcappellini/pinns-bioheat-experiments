import hydra
from omegaconf import DictConfig
import os

@hydra.main(config_path=".", config_name="config")  # Loads config.yaml from current directory
def main(cfg: DictConfig):
    # Access model properties and parameters from Hydra config
    model_props = cfg.model_properties
    model_params = cfg.model_parameters
    experiment_type = cfg.experiment.type
    n_observers = cfg.experiment.observers
    output_dir = cfg.output.base_dir

    # Example: print out the loaded configurations
    print("Model Properties:", model_props)
    print("Model Parameters:", model_params)
    print(f"Experiment Type: {experiment_type} with {n_observers} observers")
    print(f"Results will be saved in: {output_dir}")

    # Create an example output subfolder for your test
    results_dir = os.path.join(output_dir, f"test_obs_{n_observers}")
    os.makedirs(results_dir, exist_ok=True)

    # Continue with your existing model setup and experiments using the loaded properties
    # ...
    # For example, set lam, lambda from properties and parameters:
    L0 = model_props.L0
    lambda_value = model_params.lam
    print(f"Lambda (model): {L0}, Lambda (parameters): {lambda_value}")
    
    # Perform any model training or predictions
    # if experiment_type == "simulation" ... else ...
    
    # Save some results in the newly created folder
    result_path = os.path.join(results_dir, "results.txt")
    with open(result_path, "w") as f:
        f.write("Experiment results go here...\n")

if __name__ == "__main__":
    main()