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

    uu.initialize_run(cfg)
    # subprocess.run(["python", f'{src_dir}/ic_compatibility.py'])
    subprocess.run(["python", f'{src_dir}/simulation.py'])
    # subprocess.run(["python", f'{src_dir}/hpo.py'])


if __name__ == "__main__":
    main()