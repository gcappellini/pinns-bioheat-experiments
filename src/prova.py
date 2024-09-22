# import numpy as np
import os
# import plots as pp
import common as co
import utils as uu


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
models = os.path.join(git_dir, "models")



co.set_prj("3Obs_meas2")
ss = co.set_run("obs_2")

e = co.read_json(f"{ss}/properties.json")


co.find_matching_json(models, e)
