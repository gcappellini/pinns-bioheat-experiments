"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf

epochs = 50000
current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)

output_dir = f"{script_directory}/observer/"
os.makedirs(output_dir, exist_ok=True)

model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

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

def meas_data(dd, pp):
    data = np.load(f"{script_directory}/measurements/{dd}/meas_{dd}_{pp}.npz")
    x, t, exact = data["x"], data["t"], data["theta"].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    return X, y


def obs_data(dd, pp):
    data = np.load(f"{script_directory}/measurements/{dd}/observed_{dd}_{pp}.npz")
    x, t, t_0, t_1, t_bolus = data["x"], data["t"], data["t_0"], data["t_1"], data["t_bolus"]
    X = np.vstack((x, t_0, t_1, t_bolus, t)).T
    return X


def go_observer(h, alpha):

    k = 4

    # def get_initial_loss(model):
    #     model.compile("adam", lr=0.001,
    #                   )
    #     losshistory, train_state = model.train(0)
    #     return losshistory.loss_train[0]


    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=4)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        # Backend tensorflow.compat.v1 or tensorflow
        return (
            dy_t
            - alpha * dy_xx
        )


    def boundary_0(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)


    def boundary_1(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)


    def bc0_obs(x, theta, X):
        return x[:, 1:2] - theta


    def bc1_obs(x, theta, X):
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
        return dtheta_x - h*(x[:, 3:4]-x[:, 2:3]) - k * (x[:, 2:3] - theta)


    def ic_obs(x):

        y1 = x[:, 1:2]
        y2 = x[:, 2:3]
        y3 = x[:, 3:4]
        fl = h*(y3-y2)
        b1 = 6.0

        e = (fl + k * (y2 - y1))/(b1 * tf.cos(b1) + k * tf.sin(b1)) * tf.sin(b1*x[:, 0:1]) + y1
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
        # [bc_0, bc_1],
        num_domain=2560,
        num_boundary=200,
        num_initial=100,
        num_test=10000,
    )

    layer_size = [5] + [50] * 3 + [1]
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

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# def test_observer(model, date, phantom):
#     # Predictions
#     X, ttrue = meas_data(date, phantom)
#     X_obs = obs_data(date, phantom)

#     ppred = model.predict(X_obs)
#     # error_u = np.linalg.norm(theta_meas - theta_pred, 2) / np.linalg.norm(theta_meas, 2)
#     # print('Relative L2 error_u: %e' % (error_u))

#     la = len(np.unique(X[:, 0:1]))
#     le = len(np.unique(X[:, 1:]))

#     pred = ppred.reshape((le, la))
#     true = ttrue.reshape((le, la))
    
#     # Plotting
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # Plot Theta Predicted
#     im1 = axes[0].imshow(pred, cmap='inferno', aspect='auto', origin='lower',
#                          extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()], vmin=0, vmax=1)
#     axes[0].set_title('Theta Predicted')
#     axes[0].set_xlabel('X')
#     axes[0].set_ylabel('T')
#     plt.colorbar(im1, ax=axes[0])

#     # Plot Theta True
#     im2 = axes[1].imshow(true, cmap='inferno', aspect='auto', origin='lower',
#                          extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()], vmin=0, vmax=1)
#     axes[1].set_title('Theta True')
#     axes[1].set_xlabel('X')
#     axes[1].set_ylabel('T')
#     plt.colorbar(im2, ax=axes[1])

#     # Plot Difference
#     im3 = axes[2].imshow(np.abs(pred-true), cmap='inferno', aspect='auto', origin='lower',
#                          extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
#     axes[2].set_title('Difference')
#     axes[2].set_xlabel('X')
#     axes[2].set_ylabel('T')
#     plt.colorbar(im3, ax=axes[2])

#     # Save the figure
#     plt.savefig(f'{output_dir}/comparison_{date}_{phantom}.png')

#     plt.show()

def test_observer(model, phantom):
    dates = ["240124", "240125", "240125b", "240130", "240202"]
    
    fig, axes = plt.subplots(len(dates), 3, figsize=(15, 5*len(dates)))

    for i, date in enumerate(dates):
        # Predictions
        X, ttrue = meas_data(date, phantom)
        X_obs = obs_data(date, phantom)

        ppred = model.predict(X_obs)

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
    plt.savefig(f'{output_dir}/comparison_all_dates_{phantom}.png')

    plt.show()



def restore_model(model, name):

    model.restore(f"{model_dir}/model_{name}-{epochs}.ckpt", verbose=0)
    return model

name = "observer"
a = go_observer(5.46e+00, 3.89e-01)
# b = train_model(a, name)
b = restore_model(a, name)

# dates = ["240124", "240125", "240125b", "240130", "240202"]
# phantoms = ["P1"]
# for d in dates:
#     for p in phantoms:
#         test_observer(b, d, p)
test_observer(b, "P1")
# go_observer("240125", "P1", 5.46e+00, 3.89e-01)
# go_observer("240125b", "P1", 5.46e+00, 3.89e-01)
# go_observer("240130", "P1", 5.46e+00, 3.89e-01)
# go_observer("240202", "P1", 5.46e+00, 3.89e-01)







# # Animation
# def plot_animation(c):
#     filename = f"{output_dir}/animation.gif"
#     dx = 0.01  # Spatial step
#     dt = 0.01  # Time step
#     L = 1
#     t_max = 1  # Maximum time

#     # Discretization
#     x_values = np.arange(0, L, dx)
#     t_values = np.arange(0, t_max, dt)

#     u = np.zeros((len(x_values), len(t_values)))
#     p = np.zeros((len(x_values), len(t_values)))

#     # Set initial condition
#     u[:, 0] = w1(x_values)
#     XX = np.vstack((x_values, np.zeros_like(x_values))).T
#     p[:, 0] = c.predict(XX).reshape(len(XX), )

#     # Update function using exact solution
#     def update(frame):
#         x = x_values.reshape(-1, 1)
#         t = t_values[frame]
#         xt_mesh = np.concatenate([x, np.full_like(x, t)], axis=1)
#         XX = np.vstack((x_values, np.full_like(x_values, t))).T

#         u[:, frame] = summative(xt_mesh).flatten()
#         p[:, frame] = c.predict(XX).reshape(len(XX), ).flatten()

#         line.set_ydata(u[:, frame])
#         pred_line.set_ydata(p[:, frame])

#         plt.xlabel('x')
#         plt.ylabel('Amplitude')
#         plt.title(f'Wave Equation Animation at t={t:.2f}')
#         plt.legend()  # Show legend with labels
#         return line, pred_line

#     # Create the animation
#     fig, ax = plt.subplots()
#     line, = ax.plot(x_values, u[:, 0], label='Exact')  # Existing line
#     pred_line, = ax.plot(x_values, p[:, 0], label='Predicted')  # New line

#     ax.set_ylim(-1, 1)  # Adjust the y-axis limits if needed

#     ani = FuncAnimation(fig, update, frames=len(t_values), interval=1.0, blit=True)
    # plt.show()

    # # Save the animation as a GIF file
    # ani.save(filename, writer='imagemagick')