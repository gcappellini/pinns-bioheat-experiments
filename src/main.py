import subprocess
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))

@hydra.main(version_base=None, config_path=current_dir, config_name="config")

def main(cfg: DictConfig):
    # Get the experiment type from the config
    experiment_type = cfg.experiment.type

    # Path to the script directory
    src_dir = os.path.dirname(os.path.abspath(__file__))
    git_dir = os.path.dirname(src_dir)

    # Define the mapping between experiment type and script file
    script_mapping = {
        'network_test': 'network_test.py',
        'simulation': 'simulation.py',
        'measurement': f'{src_dir}/measurements.py'
    }

    # Check if the experiment type is valid
    if experiment_type in script_mapping:
        # Get the script filename
        script_to_run = script_mapping[experiment_type]
        
        # Build the full path to the script
        script_path = os.path.join(git_dir, script_to_run)
        
        # Serialize the config and pass it as a YAML file to the script
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config:
            OmegaConf.save(config=cfg, f=temp_config.name)
            temp_config_path = temp_config.name
        # Execute the script using subprocess
        subprocess.run(["python", script_path, "--config-path", temp_config_path])
    else:
        print(f"Unknown experiment type: {experiment_type}")

if __name__ == "__main__":
    main()