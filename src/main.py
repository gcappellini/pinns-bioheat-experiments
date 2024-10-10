import subprocess
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import utils as uu

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.abspath(__file__))
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")

@hydra.main(version_base=None, config_path=current_dir, config_name="config")

def main(cfg: DictConfig):
    # Get the experiment type from the config
    experiment = cfg.experiment.name
    print(f"Running experiment {experiment}...")

    cfg1 = uu.configure_settings(cfg, experiment)
    output_dir = os.path.join(tests_dir, f"{experiment[0]}_{experiment[1]}")

    OmegaConf.save(cfg1,f"{output_dir}/config.yaml")
    OmegaConf.save(cfg1,f"{src_dir}/config.yaml")

    cfg_matlab = OmegaConf.create({
    "model_properties": cfg1.model_properties,
    "model_parameters": cfg1.model_parameters,
    "experiment": cfg1.experiment.name
    })

    OmegaConf.save(cfg_matlab,f"{src_dir}/config_matlab.yaml")


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