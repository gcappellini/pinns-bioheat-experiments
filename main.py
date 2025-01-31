import subprocess
import os, logging
import hydra
from omegaconf import DictConfig, OmegaConf
from src.common import setup_logging


git_dir = os.getcwd()
src_dir = os.path.join(git_dir, "src")
conf_dir = os.path.join(src_dir, "configs")

logger = setup_logging()

def initialize_run(cfg1):

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg1.output_dir = output_dir
    OmegaConf.save(cfg1,f"{output_dir}/config.yaml")
    OmegaConf.save(cfg1,f"{conf_dir}/config_run.yaml")
    logger.info(f'Working dir: {output_dir}')

    return cfg1, output_dir

@hydra.main(version_base=None, config_path=conf_dir, config_name="config_run")
def main(cfg: DictConfig):
    
    initialize_run(cfg)
    
    subprocess.run(["python3", f'{src_dir}/simulation.py'])

if __name__ == "__main__":
    main()