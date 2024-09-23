import subprocess
import os
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Get the experiment type from the config
    experiment_type = cfg.experiment.type
    
    # Path to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the mapping between experiment type and script file
    script_mapping = {
        'network_test': 'network_test.py',
        'simulation': 'simulation.py',
        'measurement': 'measurement.py'
    }

    # Check if the experiment type is valid
    if experiment_type in script_mapping:
        # Get the script filename
        script_to_run = script_mapping[experiment_type]
        
        # Build the full path to the script
        script_path = os.path.join(script_dir, script_to_run)
        
        # Execute the script using subprocess
        subprocess.run(["python", script_path])
    else:
        print(f"Unknown experiment type: {experiment_type}")

if __name__ == "__main__":
    main()