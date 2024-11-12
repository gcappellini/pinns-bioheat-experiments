import subprocess
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import utils as uu

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.abspath(__file__))
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

@hydra.main(version_base=None, config_path=current_dir, config_name="config")
def main(cfg: DictConfig):

    experiment = cfg.experiment.name
    print(f"Running experiment {experiment}...")
    cfg1 = uu.configure_settings(cfg, experiment)
    uu.initialize_run(cfg1)

    # print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    if cfg.experiment.import_data:
        subprocess.run(["python", f'{src_dir}/import_data.py'])

    if experiment[1].startswith("simulation"): 
        subprocess.run(["python", f'{src_dir}/simulation.py'])

    if experiment[1].startswith("meas_"):    
        subprocess.run(["python", f'{src_dir}/measurements.py'])


if __name__ == "__main__":
    main()