import deepxde as dde
import numpy as np
from scipy import integrate
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from scipy.interpolate import interp1d
import matplotlib.cm as cm


current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)
output_dir = f"{script_directory}/multiple_observer"
os.makedirs(output_dir, exist_ok=True)

model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

figures_dir = os.path.join(output_dir, "figures")
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

epochs = 50000
K = 4
L_0 = 0.15

rho, cp, k = 1000, 4181, 0.563
alpha = k/(rho*cp)

experiments = ["AX1", "AX2", "BX1", "BX2", "AY1", "AY2", "BY1", "BY2"]


def meas_data(date):
    data = np.load(f"{script_directory}/measurements/meas_{date}.npz")
    x, t, exact = data["x"], data["t"], data["theta"].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    return X, y


def obs_data(date):
    data = np.load(f"{script_directory}/measurements/obs_{date}.npz")
    x, t, y_0, y_1, y_2 = data["x"], data["t"], data["t_0"], data["t_1"], data["t_bolus"]
    X = np.vstack((x, y_0, y_1, y_2, t)).T
    return X


def get_initial_loss(model):
    model.compile("adam", lr=0.001,
                    )
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]



def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def bc0_obs(x, theta, X):
    return x[:, 1:2] - theta


def configure_subplot(ax, XS, surface):
    la = len(np.unique(XS[:, 0:1]))
    le = len(np.unique(XS[:, 1:]))
    X = XS[:, 0:1].reshape(le, la)
    T = XS[:, 1:].reshape(le, la)

    ax.plot_surface(X, T, surface, cmap='inferno', alpha=.8)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.dist = 10
    ax.view_init(20, -120)

    # Set axis labels
    ax.set_xlabel('Depth', fontsize=7, labelpad=-1)
    ax.set_ylabel('Time', fontsize=7, labelpad=-1)
    ax.set_zlabel('Theta', fontsize=7, labelpad=-4)


def create_observer(h):
    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=4)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        # Backend tensorflow.compat.v1 or tensorflow
        return (
            dy_t
            - alpha * dy_xx
        )

    def bc1_obs(x, theta, X):
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
        return dtheta_x - h*(x[:, 3:4]-x[:, 2:3]) - K * (x[:, 2:3] - theta)


    def ic_obs(x):

        z = x[:, 0:1]
        y1 = x[:, 1:2]
        y2 = x[:, 2:3]
        y3 = x[:, 3:4]
        beta = h * (y3 - y2) + K * (y2 -y1)
        a2 = 1.0

        e = y1 + ((beta - ((2/L_0)+K)*a2)/((1/L_0)+K))*z + a2*z**2
        return e

    xmin = [0, 0, 0, 0]
    xmax = [1, 1, 1, 1]
    geom = dde.geometry.Hypercube(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.OperatorBC(geomtime, bc0_obs, boundary_0)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)


    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_0, bc_1, ic],
        num_domain=2560,
        num_boundary=200,
        num_initial=100,
        num_test=10000,
    )

    layer_size = [5] + [150] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)

    # initial_losses = get_initial_loss(model)
    # loss_weights = len(initial_losses) / initial_losses

    model.compile(
        "adam", lr=0.001,
        # loss_weights=loss_weights
    )
    return model

def train_model(model, name):
    losshistory, train_state = model.train(iterations=epochs, model_save_path=f"{model_dir}/model_{name}")
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, loss_fname=f"loss_{name}.dat", train_fname=f"train_{name}.dat", test_fname=f"test_{name}.dat", output_dir=f"{model_dir}")

    return model


def restore_model(model, name):

    checkpoint_path = f"{model_dir}/model_{name}-{epochs}.pt"
    if os.path.exists(checkpoint_path):
        model.restore(f"{model_dir}/model_{name}-{epochs}.pt", verbose=0)
        print(f"Model {model, name} restored from checkpoint.")
    else:
        train_model(model, name)
    return model


def mu(o, tau, d, p):
    XO = obs_data(d, p)
    instants = np.unique(XO[:, 4:5])
    # instants = XO[:, 4:5]
    XO_all = XO[XO[:, 0]==np.max(XO[:, 0])]

    y1 = XO_all[:, 1:2].reshape(len(instants),)
    f1 = interp1d(instants, y1, kind='previous')

    y2 = XO_all[:, 2:3].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    y3 = XO_all[:, 3:4].reshape(len(instants),)
    f3 = interp1d(instants, y3, kind='previous')

    tm = 0.996268656716418
    if tau > tm:
        tau = tm

    XOt = np.vstack((np.max(XO_all[:, 0]), f1(tau), f2(tau), f3(tau), tau)).T
    th = f2(tau)
    muu = []
    for el in o.values():
        oss = el.predict(XOt)
        scrt = np.abs(oss-th)
        muu.append(scrt)
    muu = np.array(muu).reshape(len(muu),)

    return muu


def plot_weights():
    lam_values = [10, 200, 1000]
    dates = ["240124", "240125", "240125b", "240130", "240202"]
    num_dates = len(dates)

    fig, axs = plt.subplots(num_dates, len(lam_values), figsize=(15, 5*num_dates))

    for row, date in enumerate(dates):
        for col, lam in enumerate(lam_values):
            a = np.load(f'{output_dir}/weights/weights_{date}_lambda_{lam}.npy', allow_pickle=True)      
            t = a[0]
            x = a[1:]
            
            cmap = cm.get_cmap('tab10' if x.shape[0] <= 10 else 'viridis')  
            ax = axs[row, col]
            for j in range(x.shape[0]):
                color = cmap(j)
                ax.plot(t, x[j], alpha=1.0, linewidth=1.8, color=color, label=f"$p_{j+1}$")

            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel(xlabel=r"Time t")
            ax.set_ylabel(ylabel=r"Weights $p_j$")
            ax.legend()
            ax.set_title(f"Dynamic weights, $\lambda={lam}$", weight='semibold')
            ax.grid()
            
            if col == 0:
                ax.annotate(f"Experiment {date}", xy=(0, 1), xytext=(-20, 20),
                            xycoords='axes fraction', textcoords='offset points',
                            fontsize=12, ha='right', va='bottom', weight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/weights_ode.png", dpi=150, bbox_inches='tight')
    plt.show()


# def plot_observers_tf(multi_obs, lam, results_dir, d, p):
#     a = np.load(f'{results_dir}/weights_lambda_{lam}.npy', allow_pickle=True)
#     y = a[1:]
#     w_f = y[:, -1]

#     XO = obs_data(d, p)

#     XOf = XO[XO[:, 4] == np.max(XO[:, 4])]
#     x = XOf[:, 0:1]

#     _, stot = meas_data(d, p)
#     sf = stot[-len(XOf):, :]

#     observer_predictions = [modelu.predict(XOf) for modelu in multi_obs.values()]
    
#     o8f = sum(w * o for w, o in zip(w_f, observer_predictions))

#     fig = plt.figure()
#     ax2 = fig.add_subplot(111)
#     cmap = cm.get_cmap('tab10' if x.shape[0] <= 10 else 'viridis')
    
#     ax2.plot(x, sf, linestyle='None', marker="o", mfc='none', markersize=8, markeredgewidth=1.5, color='C0', label="System")

#     for i, modelu in enumerate(multi_obs.values()):
#         color = cmap(i)
#         ax2.plot(x, observer_predictions[i], alpha=1.0, linewidth=1.8, color=color, label=f"Observer {i+1}")
    

#     ax2.plot(x, o8f, linestyle='None', marker="X", markersize=7, color='gold', label="MM adaptive observer")

#     ax2.set_xlim(0, 1)
#     plt.xticks(np.arange(0, 1.01, 0.1))
#     ax2.legend()
#     ax2.set_ylabel(ylabel=r"Temperature")
#     ax2.set_xlabel(xlabel=r"Distance $x$")
#     ax2.set_title(r"Solutions at $\tau=t_f$", weight='semibold')
#     plt.grid()

#     plt.savefig(f"{results_dir}/tf_observers_{lam}.png", dpi=150, bbox_inches='tight')
#     plt.show()

#     plt.close(fig)


def plot_observer_l2(grid, pred, truth, name, dd):
    together = np.concatenate((grid, pred, truth), axis=1)
    # together[:, 0] = x, together[:, 1] = y1, together[:, 2] = y2, together[:, 3] = y3, together[:, 4] = t, together[:, 5] = pred, together[:, 6] = truth
    l2_k = []
    tt = np.unique(together[:, 4])
    for te in tt:
        tm = 0.9990108803165183
        if te > tm:
            te = tm
        
        # Select the corresponding row from XO
        XOt = together[together[:, 4] == te]
        pr = XOt[:, 5]
        tr = XOt[:, 6]

        l2_k.append(dde.metrics.l2_relative_error(pr, tr))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(tt, l2_k, alpha=1.0, linewidth=1.8, color='C0')

    ax1.set_xlabel(xlabel=r"Time t", fontsize=7)  # xlabel
    ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)  # ylabel
    ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
    # ax1.set_ylim(bottom=0.0)
    # ax1.set_xlim(0, 1.01)
    # plt.yticks(fontsize=7)

    plt.grid()
    plt.savefig(f"{figures_dir}/l2_observer_{name}_{dd}.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.close(fig)


def mm_ode(multi_obs, pp):
    weights_dir = os.path.join(output_dir, "weights")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    dates = ["240124", "240125", "240125b", "240130", "240202"]
    ej = len(multi_obs)
    p0 = np.full((ej,), 1/ej)

    lem = [10, 200, 1000]

    for dd in dates:
        for lam in lem:
            def f(t, p):
                a = mu(multi_obs, t, dd, pp)
                e = np.exp(-1*a)
                d = np.inner(p, e)
                f = []
                for el in range(len(p)):
                    ccc = - lam * (1-(e[el]/d))*p[el]
                    f.append(ccc)
                return np.array(f)


            sol = integrate.solve_ivp(f, (0, 1), p0, t_eval=np.linspace(0, 1, 100))
            y = sol.y
            t = sol.t
            weights = np.zeros((sol.y.shape[0]+1, sol.y.shape[1]))
            weights[0] = sol.t
            weights[1:] = sol.y
            np.save(f'{weights_dir}/weights_{dd}_lambda_{lam}.npy', weights)


def all_l2_errors(model):

    observer_numbers = range(1, len(model) + 1)
    hh_values = model.keys()

    df = pd.DataFrame({'Obs #': observer_numbers, 'h': hh_values})

    for dd in experiments:
        PO = obs_data(dd)
        _, tmeas = meas_data(dd)
        # PO = np.concatenate((PO, tmeas), axis=1)

        l2_errors = []
        
        for key, modelu in model.items():
            e = modelu.predict(PO)
            l2_err = dde.metrics.l2_relative_error(e, tmeas)
            l2_errors.append(l2_err)
            plot_observer_l2(PO, e, tmeas, key, dd)

        df[f'{dd}'] = l2_errors

    # Specify the columns over which to compute the L2 norm
    columns_to_include = experiments

    # Calculate the L2 norm across the specified columns for each row
    df['Overall error'] = np.linalg.norm(df[columns_to_include], axis=1)
    df.to_excel(f'{output_dir}/l2_errors.xlsx', index=False)


def mm_predict(m_obs, da, xob):
    la = 1000
    a = np.load(f'{output_dir}/weights/weights_{da}_lambda_{la}.npy', allow_pickle=True) 

    num_observers = len(multi_obs)
    tt = xob[:, 4:]

    # for te in range(len(tt)):
    for te in tt:
        tm = 0.9990108803165183
        if te > tm:
            te = tm
        
        # Find the index of the value in a[0] closest to tt[te]
        idx_closest = np.where(np.isclose(a[0], te))[0]
        

        observer_predictions = [modelu.predict(xob) for modelu in m_obs.values()]
        # if te == tm:
        #     te = 1
        o = sum(a[i + 1, idx_closest] * observer_predictions[i] for i in range(num_observers))
        return o




def plot_mm_observer(model, phantom):
    dates = ["240124", "240125", "240125b", "240130", "240202"]
    
    fig, axes = plt.subplots(len(dates), 3, figsize=(15, 5*len(dates)))

    for i, date in enumerate(dates):
        # Predictions
        X, ttrue = meas_data(date, phantom)
        X_obs = obs_data(date, phantom)

        ppred = mm_predict(model, date, X_obs)

        la = len(np.unique(X[:, 0:1]))
        le = len(np.unique(X[:, 1:]))

        pred = ppred.reshape((le, la))
        true = ttrue.reshape((le, la))

        # Plot Theta Predicted
        im1 = axes[i, 0].imshow(pred, cmap='inferno', aspect='auto', origin='lower',
                             extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()], vmin=0, vmax=1)
        axes[i, 0].set_title(f'Theta Predicted ({date})')
        axes[i, 0].set_xlabel('X')
        axes[i, 0].set_ylabel('T')
        plt.colorbar(im1, ax=axes[i, 0])

        # Plot Theta True
        im2 = axes[i, 1].imshow(true, cmap='inferno', aspect='auto', origin='lower',
                             extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()], vmin=0, vmax=1)
        axes[i, 1].set_title(f'Theta True ({date})')
        axes[i, 1].set_xlabel('X')
        axes[i, 1].set_ylabel('T')
        plt.colorbar(im2, ax=axes[i, 1])

        # Plot Difference
        im3 = axes[i, 2].imshow(np.abs(pred-true), cmap='inferno', aspect='auto', origin='lower',
                             extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
        axes[i, 2].set_title(f'Difference ({date})')
        axes[i, 2].set_xlabel('X')
        axes[i, 2].set_ylabel('T')
        plt.colorbar(im3, ax=axes[i, 2])

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{output_dir}/comparison_mm_{phantom}.png')

    plt.show()


h_unk = np.linspace(8, 160, num=10).round(1)

multi_obs = {}

for hh in h_unk:
    modelu = create_observer(hh)
    modelu = restore_model(modelu, f"obs_{hh}")
    multi_obs[hh] = modelu

all_l2_errors(multi_obs)
# mm_ode(multi_obs, "P1")
# plot_weights()
# plot_mm_observer(multi_obs, "P1")

# plot_observers_tf(multi_obs, lam, results_dir, dd, pp)
# plot_observers_l2(multi_obs, lam, results_dir, dd, pp)
