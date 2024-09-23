import os
import utils as uu


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
models = os.path.join(git_dir, "models")


uu.mm_observer(8)