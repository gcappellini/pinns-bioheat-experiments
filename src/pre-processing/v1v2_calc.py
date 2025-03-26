import os
from omegaconf import OmegaConf
import numpy as np
from hydra import initialize, compose


current_file: str = os.path.abspath(__file__)
src_dir: str = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
models: str = os.path.join(git_dir, "models")

conf_dir: str = os.path.join(src_dir, "configs")




initialize('configs', version_base=None)
cfg = compose(config_name='config_run')
# cfg = OmegaConf.load(f"{conf_dir}/config_run.yaml")

# Calculate pdecoeff



OmegaConf.save(cfg, f"{conf_dir}/config_run.yaml")


r_cat = 0.5/1000  # Radius Cooling 1 (m)
n_cat_cooling_1 = 1.0           # n° of catethers Cooling 1
n_cat_cooling_2 = 2.0           # n° of catethers Cooling 2

Q_cooling_1_lmin = 0.4/17                    # Volumetric flow Cooling 1 (L/min)
Q_cooling_2_lmin = 1.9/17                    # Volumetric flow Cooling 2 (L/min)

Q_cooling_1 = (0.4/17)*(1/1000)*(1/60)    # Volumetric flow Cooling 1 (m^3/s)
Q_cooling_2 = (1.9/17)*(1/1000)*(1/60)    # Volumetric flow Cooling 2 (m^3/s)

A_cooling_1 = np.pi*n_cat_cooling_1*(r_cat**2)    # Area Cooling 1 (m^2)
A_cooling_2 = np.pi*n_cat_cooling_2*(r_cat**2)    # Area Cooling 2 (m^2)

v_cooling_1 = round(Q_cooling_1/A_cooling_1, 3) # Velocity Cooling 1 (m/s)
v_cooling_2 = round(Q_cooling_2/A_cooling_2, 3) # Velocity Cooling 2 (m/s)

if __name__ == "__main__":
    print(os.path.abspath(cfg.run.dir))
    # print(f"{a1} & {a2} & {a3} & {a4} & {a5}")
    # print(f"{theta10} & {theta20} & {theta30} & {theta_gt20}")
    # print(f"{b1} & {b2} & {b3} & {b4}")
    # print(f"{c1} & {c2} & {c3} & {upsilon} & {lamb}")
    print(f"{Q_cooling_1} & {v_cooling_1}")
    print(f"{Q_cooling_2} & {v_cooling_2}")