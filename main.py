import subprocess
import os, logging
import hydra
from omegaconf import DictConfig, OmegaConf


git_dir = os.getcwd()
src_dir = os.path.join(git_dir, "src")
conf_dir = os.path.join(src_dir, "configs")
# git_dir = os.path.dirname(src_dir)
# tests_dir = os.path.join(git_dir, "tests")
# os.makedirs(tests_dir, exist_ok=True)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=conf_dir, config_name="config_run")
def main(cfg: DictConfig):

    logger.info(f'Working dir: {os.getcwd()}')



if __name__ == "__main__":
    main()