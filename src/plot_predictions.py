import utils as uu
import os
import common as co
import numpy as np
import wandb
import plots as pp
import matplotlib.pyplot as plt
from scipy import integrate
from import_vessel_data import load_measurements, extract_entries, scale_df

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

# PINNs implementation
prj_figs = co.set_prj("test_cooling")

multi_obs = uu.mm_observer()

a = uu.import_testdata()
X = a[:, 0:2]
meas = a[:, 2:3]
bolus = a[:, 3:4]

x_obs = uu.import_obsdata()

# for el in range(len(multi_obs)):
#     run_figs = co.set_run(f"obs_{el}")
#     aa = uu.read_json(f"{run_figs}/properties.json")
    # wandb.init(
    # project="test_meas", name=f"obs_{el}",
    # config=aa
    # )
    # pred = multi_obs[el].predict(x_obs)
    # meas = meas.reshape(pred.shape)
    # pp.check_obs(X, y_obs[:, el], pred, el, run_figs)
    # pp.plot_l2_tf(X, meas, pred, multi_obs[el], el, run_figs)
    # b = uu.compute_metrics(meas, pred)
    # wandb.log(b)
    # wandb.finish()

# t = np.unique(X[:, 1:2])
# mu = uu.mu(multi_obs, t)
# pp.plot_mu(mu, t, prj_figs)

n_obs = 8

x = uu.gen_obsdata()

p0 = np.full((n_obs,), 1/n_obs)
par = uu.read_json(f"{src_dir}/parameters.json")
lam = par["lambda"]

def f(t, p):
    a = uu.mu(multi_obs, t)
    e = np.exp(-1*a)
    d = np.inner(p, e)
    f = []
    for el in range(len(p)):
        ccc = - lam * (1-(e[:, el]/d))*p[el]
        f.append(ccc)
    return np.array(f).reshape(len(f),)


sol = integrate.solve_ivp(f, (0, 1), p0, t_eval=np.linspace(0, 1, 100))
x = sol.y
t = sol.t
weights = np.zeros((sol.y.shape[0]+1, sol.y.shape[1]))
weights[0] = sol.t
weights[1:] = sol.y
np.save(f'{prj_figs}/weights_lam_{lam}.npy', weights)
pp.plot_weights(weights[1:], weights[0], prj_figs)

mm_obs_pred = uu.mm_predict(multi_obs, lam, x_obs, prj_figs)
preds = np.vstack((x_obs[:, 0], x_obs[:, -1], mm_obs_pred)).T

positions = uu.get_tc_positions()
y2_pred_sc = preds[preds[:, 0] == positions[0]][:, 2]
gt2_pred_sc = preds[preds[:, 0] == positions[1]][:, 2]
gt1_pred_sc = preds[preds[:, 0] == positions[2]][:, 2]
y1_pred_sc = preds[preds[:, 0] == positions[3]][:, 2]

y2_pred = uu.rescale_t(y2_pred_sc)
gt2_pred = uu.rescale_t(gt2_pred_sc)
gt1_pred = uu.rescale_t(gt1_pred_sc)
y1_pred = uu.rescale_t(y1_pred_sc)

pp.plot_l2_tf(X, meas, preds[:, -1], multi_obs, 0, prj_figs, MultiObs=True)

# Plot predictions on the 2D
file_path = f"{src_dir}/data/vessel/20240522_1.txt"
timeseries_data = load_measurements(file_path)
df = extract_entries(timeseries_data, 83*60, 4*60*60)

fig, ax = plt.subplots(figsize=(12, 6))  # Stretching layout horizontally

# Plotting data with specified attributes
ax.plot(df['t']/60, df['y1'], alpha=0.8, linestyle="--")
ax.plot(df['t']/60, df['gt1'], alpha=0.8, linestyle="--", linewidth=0.7)
ax.plot(df['t']/60, df['gt2'], alpha=0.8, linestyle="--", linewidth=0.7)
ax.plot(df['t']/60, df['y2'], alpha=0.8, linestyle="--", linewidth=0.7)
ax.plot(df['t']/60, df['y3'], alpha=1.0, label="y3", linewidth=0.7)

ax.plot(df['t']/60, y1_pred, label='y1', color="C0", linewidth=0.7)
ax.plot(df['t']/60, gt1_pred, label='gt1', color="C1",linewidth=0.7)
ax.plot(df['t']/60, gt2_pred, label='gt2', color="C2",linewidth=0.7)
ax.plot(df['t']/60, y2_pred, label='y2', color="C3",linewidth=0.7)


# Add vertical dashed red lines with labels on the plot
# ax.axvline(x=2, color='red', linestyle='--', linewidth=1.1)
# ax.text(2.5, 35.8, 'RF on,\nmax perfusion', color='red', fontsize=10, verticalalignment='top')

# ax.axvline(x=29, color='red', linestyle='--', linewidth=1.1)
# ax.text(29.5, 35.8, 'Min perfusion', color='red', fontsize=10, verticalalignment='top')

# ax.axvline(x=56, color='red', linestyle='--', linewidth=1.1)
# ax.text(56.5, 35.8, 'Zero perfusion', color='red', fontsize=10, verticalalignment='top')

# ax.axvline(x=83, color='red', linestyle='--', linewidth=1.1)
# ax.text(83.5, 35.8, 'RF off, max perfusion', color='red', fontsize=10, verticalalignment='top')

# Adding legend for the plotted data (excluding the vertical lines)
ax.legend()

# Setting title and labels with modifications
ax.set_title("Cooling Experiment", fontweight='bold')
ax.set_xlabel("Time (min)", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
# ax.set_xlim(0, 234)

# Adjust layout for better horizontal stretching
plt.tight_layout()

# Display and save plot

plt.savefig(f"{prj_figs}/2D_plot.png", dpi=120)
plt.show()
plt.close()
plt.clf()


