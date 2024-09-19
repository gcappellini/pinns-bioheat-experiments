# import numpy as np
import os
# import plots as pp
import common as co
import utils as uu


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
models = os.path.join(git_dir, "models")

a = uu.gen_obsdata(8)
print(uu.rescale_t(a[0, 2]))

