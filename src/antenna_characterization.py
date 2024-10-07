import os
import utils as uu
import common as co
import coeff_calc as cc
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
models = os.path.join(git_dir, "models")


a = OmegaConf.load(f"{src_dir}/config.yaml")
co.set_prj("antenna_characterization")

b = uu.load_from_pickle(f"{src_dir}/data/vessel/antenna_characterization.pkl")
real_b = uu.rescale_df(b)


x = uu.rescale_x(uu.get_tc_positions())
labels = ['y2', 'gt2', 'gt1', 'y1']

SAR = {}
columns = ["x", "SAR"]
df = pd.DataFrame(columns=columns)


for el in range(len(x)):
    point=labels[el]
    TR_t1 = (real_b[point][1]-real_b[point][0])/(real_b['t'][1]/60)
    TR_t2 = (real_b[point][2]-real_b[point][0])/(real_b['t'][2]/60)
    TR_t3 = (real_b[point][3]-real_b[point][0])/(real_b['t'][3]/60)
    SAR_pt = {"x": [x[el], x[el], x[el]],
              "SAR": [60*TR_t1, 60*TR_t2, 60*TR_t3]}
    df_pt = pd.DataFrame(SAR_pt)
    df = df._append(df_pt)
    mean_SAR = df_pt.groupby('x')['SAR'].mean()

xlabel, _, _ = uu.get_scaled_labels(True)

x_formula = np.linspace(x[0], x[-1], num=100)
SAR_formula = uu.SAR(x_formula)


# Plot
fig, ax = plt.subplots()

ax.plot(x_formula, SAR_formula, label="Model", color="red", linestyle="--")
# ax.errorbar(df["x"], df["SAR"], label="measured", yerr=0.4)


# plt.figure(figsize=(8, 6))
sns.lineplot(x='x', y='SAR', data=df, marker='o', 
             estimator=np.mean, errorbar='sd', ax=ax, label='Measured SAR')

plt.legend()
plt.title('Measured SAR with Standard Deviation')
plt.xlabel(xlabel)
plt.ylabel('SAR (W/kg)')
plt.grid()
plt.show()