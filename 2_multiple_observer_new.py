import deepxde as dde
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import seaborn as sns


jj =  [100, 500, 10000, 20]
# K = 100
for K in jj:
    current_file = os.path.abspath(__file__)
    script_directory = os.path.dirname(current_file)
    output_dir = f"{script_directory}/multiple_observer_k{K}"
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    figures_dir = os.path.join(output_dir, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    weights_dir = os.path.join(output_dir, "weights")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    meas_dir = f"{script_directory}/measurements"

    epochs = 50000
    L_0 = 0.15

    rho, cp, k = 1000, 4181, 0.563
    alpha = k/(rho*cp)

    labels = [["AX1", "AX2", "BX1", "BX2"], ["AY1", "AY2", "BY1", "BY2"]]


    def flatten_ext(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list

    def meas_data(date):
        data = np.load(f"{meas_dir}/meas_{date}.npz")
        x, t, exact = data["x"], data["t"], data["theta"].T
        X = np.vstack((x, t)).T
        y = exact.flatten()[:, None]
        return X, y


    def obs_data(date):
        data = np.load(f"{meas_dir}/obs_{date}.npz")
        x, t, y_0, y_1, y_2 = data["x"], data["t"], data["t_0"], data["t_1"], data["t_bolus"]
        X = np.vstack((x, y_0, y_1, y_2, t)).T
        return X


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

        model.compile(
            "adam", lr=0.001,
        )
        return model


    def train_model(model, name):
        losshistory, train_state = model.train(iterations=epochs, model_save_path=f"{model_dir}/model_{name}")
        dde.saveplot(losshistory, train_state, issave=True, isplot=True, loss_fname=f"loss_{name}.dat", train_fname=f"train_{name}.dat", test_fname=f"test_{name}.dat", output_dir=f"{model_dir}")

        # Convert the list of arrays to a 2D NumPy array
        matrix = np.array(losshistory.loss_train)

        # Separate the components into different arrays
        loss_res = matrix[:, 0]
        loss_bc0 = matrix[:, 1]
        loss_bc1 = matrix[:, 2]
        loss_ic = matrix[:, 3]

        # l2_error = np.array(losshistory.metrics_test)

        fig = plt.figure(figsize=(6, 5))
        iters = 500 * np.arange(len(loss_res))
        with sns.axes_style("darkgrid"):
            plt.plot(iters, loss_res, label='$\mathcal{L}_{r}$')
            plt.plot(iters, loss_bc0, label='$\mathcal{L}_{u_0}$')
            plt.plot(iters, loss_bc1, label='$\mathcal{L}_{u_t}$')
            plt.plot(iters, loss_ic, label='$\mathcal{L}_{u_t}$')
            # plt.plot(iters, l2_error, label='$\mathcal{L}^2 error$')
            plt.yscale('log')
            plt.xlabel('iterations')
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(f"{model_dir}/losses_{name}.png")
            plt.show()
        return model


    def restore_model(model, name):
        checkpoint_path = f"{model_dir}/model_{name}-{epochs}.pt"
        if os.path.exists(checkpoint_path):
            model.restore(f"{model_dir}/model_{name}-{epochs}.pt", verbose=0)
            print(f"Model {model, name} restored from checkpoint.")
        else:
            train_model(model, name)
        return model


    def test_observer(mm, name):
        obs_dir = os.path.join(figures_dir, name)
        if not os.path.exists(obs_dir):
            os.makedirs(obs_dir)

        # Create figure 1
        fig1, axs1 = plt.subplots(2, 2, figsize=(13, 7))

        # Load and plot data for figure 1
        for j, label in enumerate(labels[0]):
            x_sys, theta = meas_data(label)
            x = x_sys[:, 0]
            t = x_sys[:, 1]
            x_obs = obs_data(label)
            theta_pred = mm.predict(x_obs)
            diff = np.abs(theta_pred - theta)

            la = len(np.unique(x))
            le = len(np.unique(t))

            # Plot theta vs x and t using imshow
            axs1[j//2, j%2].scatter(x, t, c=diff, cmap='inferno', s=60, marker='o', edgecolors='none')
            axs1[j//2, j%2].set_title(f"{label}", fontweight="bold")
            axs1[j//2, j%2].set_xlabel('Z', fontsize=12)
            axs1[j//2, j%2].set_ylabel(r'$\tau$', fontsize=12)
            plt.colorbar(axs1[j//2, j%2].scatter([], [], c=[], cmap='inferno'), ax=axs1[j//2, j%2], label=r'$|\theta_{pred} - \theta_{true}|$')


        # Adjust layout and save figure 1
        plt.tight_layout()
        plt.savefig(f'{obs_dir}/error_X.png')
        plt.close()

        # Create figure 2
        fig2, axs2 = plt.subplots(2, 2, figsize=(13, 7))

        # Load and plot data for figure 2
        for j, label in enumerate(labels[1]):
            x_sys, theta = meas_data(label)
            x = x_sys[:, 0]
            t = x_sys[:, 1]
            x_obs = obs_data(label)
            theta_pred = mm.predict(x_obs)
            diff = np.abs(theta_pred - theta)

            la = len(np.unique(x))
            le = len(np.unique(t))

            axs2[j//2, j%2].scatter(x, t, c=diff, cmap='inferno', s=60, marker='o', edgecolors='none')
            axs2[j//2, j%2].set_title(f"{label}", fontweight="bold")
            axs2[j//2, j%2].set_xlabel('Z', fontsize=12)
            axs2[j//2, j%2].set_ylabel(r'$\tau$', fontsize=12)
            plt.colorbar(axs2[j//2, j%2].scatter([], [], c=[], cmap='inferno'), ax=axs2[j//2, j%2], label=r'$|\theta_{pred} - \theta_{true}|$')

        # Adjust layout and save figure 2
        plt.tight_layout()
        plt.savefig(f'{obs_dir}/error_Y.png')
        plt.close()

        # Create figure 3
        fig3, axs3 = plt.subplots(2, 2, figsize=(13, 7))

        # Load and plot data for figure 3
        for i, label in enumerate(labels[0]):
            x_sys, theta = meas_data(label)
            x = x_sys[:, 0]
            t = x_sys[:, 1]
            x_obs = obs_data(label)
            theta_pred = mm.predict(x_obs)
            together = np.concatenate((x_obs, theta_pred, theta), axis=1)
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

            # Plot t_0, t_1, and t_bolus against t on each subplot
            axs3[i//2, i%2].plot(tt, l2_k, 'r-', label=r'$L^2$ norm')
            axs3[i//2, i%2].set_xlabel(r"$\tau$", fontsize=12)
            axs3[i//2, i%2].set_ylabel(r"$L^2$ norm", fontsize=12)
            axs3[i//2, i%2].set_title(f"{label}", fontsize=14, fontweight="bold")
            axs3[i//2, i%2].tick_params(axis='both', which='major', labelsize=10)
            axs3[i//2, i%2].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig(f'{obs_dir}/l2_X.png')
        plt.show()
        plt.close()


        # Create figure 4
        fig4, axs4 = plt.subplots(2, 2, figsize=(13, 7))

        # Load and plot data for figure 4
        for i, label in enumerate(labels[1]):
            x_sys, theta = meas_data(label)
            x = x_sys[:, 0]
            t = x_sys[:, 1]
            x_obs = obs_data(label)
            theta_pred = mm.predict(x_obs)
            together = np.concatenate((x_obs, theta_pred, theta), axis=1)
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

            # Plot observing['t_inf'] vs observing['time'] on the left subplot
            axs4[i//2, i%2].plot(tt, l2_k, 'r-', label=r'$L^2$ norm')
            axs4[i//2, i%2].set_xlabel(r"$\tau$", fontsize=12)
            axs4[i//2, i%2].set_ylabel(r"$L^2$ norm", fontsize=12)
            axs4[i//2, i%2].set_title(f"{label}", fontsize=14, fontweight="bold")
            axs4[i//2, i%2].tick_params(axis='both', which='major', labelsize=10)
            axs4[i//2, i%2].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig(f'{obs_dir}/l2_Y.png')
        plt.show()
        plt.close()


    def mu(o, tau, n):
        XO = obs_data(n)
        instants = np.unique(XO[:, 4])
        # instants = XO[:, 4:5]
        XO_all = XO[XO[:, 0]==np.max(XO[:, 0])]

        y1 = XO_all[:, 1:2].reshape(len(instants),)
        f1 = interp1d(instants, y1, kind='previous')

        y2 = XO_all[:, 2:3].reshape(len(instants),)
        f2 = interp1d(instants, y2, kind='previous')

        y3 = XO_all[:, 3:4].reshape(len(instants),)
        f3 = interp1d(instants, y3, kind='previous')

        tm = 0.9957446808510638
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


    def mm_ode(multi_obs, lem):

        ej = len(multi_obs)
        p0 = np.full((ej,), 1/ej)

        for dd in flatten_ext(labels):
            for lam in lem:
                def f(t, p):
                    a = mu(multi_obs, t, dd)
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


    def plot_weights(lam_values):

        fig, axs = plt.subplots(len(labels[0]), len(lam_values), figsize=(15, 5*len(labels[0])))

        for row, dd in enumerate(labels[0]):
            for col, lam in enumerate(lam_values):
                a = np.load(f'{weights_dir}/weights_{dd}_lambda_{lam}.npy', allow_pickle=True)      
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
                    ax.annotate(f"{dd}", xy=(0, 1), xytext=(-20, 20),
                                xycoords='axes fraction', textcoords='offset points',
                                fontsize=12, ha='right', va='bottom', weight='bold')

        plt.tight_layout()
        plt.savefig(f"{figures_dir}/weights_X.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        plt.clf()


        fig1, axs1 = plt.subplots(len(labels[1]), len(lam_values), figsize=(15, 5*len(labels[1])))

        for row, dd in enumerate(labels[1]):
            for col, lam in enumerate(lam_values):
                a = np.load(f'{weights_dir}/weights_{dd}_lambda_{lam}.npy', allow_pickle=True)      
                t = a[0]
                x = a[1:]
                
                cmap = cm.get_cmap('tab10' if x.shape[0] <= 10 else 'viridis') 
                ax = axs1[row, col]
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
                    ax.annotate(f"{dd}", xy=(0, 1), xytext=(-20, 20),
                                xycoords='axes fraction', textcoords='offset points',
                                fontsize=12, ha='right', va='bottom', weight='bold')

        plt.tight_layout()
        plt.savefig(f"{figures_dir}/weights_Y.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    h_unk = np.linspace(8, 160, num=10).round(1)

    multi_obs = {}

    for hh in h_unk:
        modelu = create_observer(hh)
        modelu = restore_model(modelu, f"obs_{hh}")
        test_observer(modelu, f"obs_{hh}")
        multi_obs[hh] = modelu

    lambdas = [10, 200, 1000]
    # all_l2_errors(multi_obs)
    mm_ode(multi_obs, lambdas)
    plot_weights(lambdas)
    # plot_mm_observer(multi_obs, "P1")

    # plot_observers_tf(multi_obs, lam, results_dir, dd, pp)
    # plot_observers_l2(multi_obs, lam, results_dir, dd, pp)
