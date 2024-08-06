import matlab.engine
import os
import matplotlib.pyplot as plt
import numpy as np

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)

eng = matlab.engine.start_matlab()
eng.cd(src_dir, nargout=0)
eng.simple_script(nargout=0)
eng.quit()


