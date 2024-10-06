import os
import utils as uu
import common as co
import coeff_calc as cc
from omegaconf import OmegaConf


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
models = os.path.join(git_dir, "models")


a = OmegaConf.load(f"{src_dir}/config.yaml")
co.set_prj("antenna_characterization")