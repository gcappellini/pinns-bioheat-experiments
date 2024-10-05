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
y1_0 = a.heating_meas_3.y1_0
y2_0 = a.heating_meas_3.y2_0
y3_0 = a.heating_meas_3.y3_0
K = a.model_properties.K
a5 = cc.a5

# Solve the system
b_2, b_3, b_4 = uu.solve_ic_comp(uu.scale_t(y1_0), uu.scale_t(y2_0), uu.scale_t(y3_0), K, a5)
print(uu.scale_t(y2_0))
# Print the solution
print(f"b_2: {b_2}")
print(f"b_3: {b_3}")
print(f"b_4: {b_4}")