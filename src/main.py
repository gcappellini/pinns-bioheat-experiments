import subprocess
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import utils as uu

current_dir = os.path.dirname(os.path.abspath(__file__))

@hydra.main(version_base=None, config_path=current_dir, config_name="config")

def main(cfg: DictConfig):
    # Get the experiment type from the config
    experiment = cfg.experiment.name

    # Path to the script directory
    src_dir = os.path.dirname(os.path.abspath(__file__))

    if cfg.experiment.import_data:
        subprocess.run(["python", f'{src_dir}/import_data.py'])


    if cfg.experiment.run_matlab:      
        subprocess.run(["python", f'{src_dir}/ground_truth.py'])

    if cfg.experiment.run_simulation:      
        subprocess.run(["python", f'{src_dir}/simulation.py'])

    if cfg.experiment.run_measurement:      
        subprocess.run(["python", f'{src_dir}/measurements.py'])


if __name__ == "__main__":
    main()