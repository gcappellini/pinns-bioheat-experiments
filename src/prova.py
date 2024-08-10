import matlab.engine
import os
import matplotlib.pyplot as plt
import numpy as np
import simulation as ss

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)

# eng = matlab.engine.start_matlab()
# eng.cd(src_dir, nargout=0)
# eng.simple_script(nargout=0)
# eng.quit()

tw1 = ss.calculate_tw1(-0.25, 0.0005)
print(tw1)

L=0.5

