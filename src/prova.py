import os
import utils as uu
import common as co


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
models = os.path.join(git_dir, "models")
co.set_prj("training")

uu.mm_observer(8)