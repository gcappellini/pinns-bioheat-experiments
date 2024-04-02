"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import interp1d

h = dde.Variable(1.0)
alpha = dde.Variable(1.0)

epochs = 30000

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)

sub = ""

nn = ["240124", "240125", "240125b", "240130", "240202"]
pp = ["P1", "P2"]


os.makedirs(f"{script_directory}/identification{sub}", exist_ok=True)
file_path = f"{script_directory}/identification{sub}/var_eval.txt"

plt.clf()
fig, axes = plt.subplots(len(nn), len(pp), figsize=(15, 10))
fig.tight_layout(pad=3.0)

with open(file_path, "w") as log_file:
    # for date in nn:
    #     for phantom in pp:
    for i in range(len(nn)):
        for j in range(len(pp)):

            name = f"{nn[i]}_{pp[j]}"
            output_dir = f"{script_directory}/identification{sub}/{name}"

            os.makedirs(output_dir, exist_ok=True)

            # def meas_data():
            #     data = np.load(f"{script_directory}/measurements/{name}/meas_{name}.npz")
            #     x, t, exact = data["x"], data["t"], data["theta"].T
            #     X = np.vstack((x, t)).T
            #     y = exact.flatten()[:, None]
            #     return X, y


            # def obs_data():
            #     data = np.load(f"{script_directory}/measurements/{name}/observed_{name}.npz")
            #     x, t, _, _, t_bolus = data["x"], data["t"], data["t_0"], data["t_1"], data["t_bolus"]
            #     X = np.vstack((x, t, t_bolus)).T
            #     return X

            # XO = obs_data()
            # instants = np.unique(XO[:, 1:2])
            # # instants = XO[:, 4:5]
            # XO_all = XO[XO[:, 0]==np.max(XO[:, 0])]

            # data1 = XO_all[:, 2:3].reshape(len(instants),)
            # f1 = interp1d(instants, data1, kind='previous')


            # # def bolus(tau):
            # #     tm = 0.996268656716418
            # #     if tau > tm:
            # #         tau = tm            
            # #     return f1(tau)


            # def get_initial_loss(model):
            #     model.compile("adam", lr=0.001,
            #                 )
            #     losshistory, train_state = model.train(0)
            #     return losshistory.loss_train[0]


            # def pde(x, y):
            #     dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            #     dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            #     # Backend tensorflow.compat.v1 or tensorflow
            #     return (
            #         dy_t
            #         - alpha * dy_xx
            #     )


            # def boundary_r(x, on_boundary):
            #     return on_boundary and np.isclose(x[0], 1)
            

            # geom = dde.geometry.Interval(0, 1)
            # timedomain = dde.geometry.TimeDomain(0, 1)
            # geomtime = dde.geometry.GeometryXTime(geom, timedomain)

            # bc = dde.icbc.RobinBC(geomtime, lambda X, y: h * (torch.Tensor(f1(X[:, 1:2])) - y), boundary_r)
            # # ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

            # ob_x, ob_y = meas_data()
            # observe_y = dde.icbc.PointSetBC(ob_x, ob_y, component=0)

            # data = dde.data.TimePDE(
            #     geomtime,
            #     pde,
            #     [bc, observe_y],
            #     num_domain=0,
            #     num_boundary=200,
            #     num_initial=100,
            #     anchors=ob_x,
            #     num_test=10000,
            # )

            # layer_size = [2] + [150] * 3 + [1]
            # activation = "tanh"
            # initializer = "Glorot uniform"
            # net = dde.nn.FNN(layer_size, activation, initializer)

            # model = dde.Model(data, net)

            # # initial_losses = get_initial_loss(model)
            # # loss_weights = len(initial_losses) / initial_losses

            # model.compile(
            #     "adam", lr=0.001, external_trainable_variables=[h, alpha],
            #     # loss_weights=loss_weights
            # )
            # variable = dde.callbacks.VariableValue([h, alpha], period=100, filename=f"{output_dir}/var_{name}.dat")
            # losshistory, train_state = model.train(iterations=epochs, callbacks=[variable], model_save_path=f"{output_dir}/model_{name}")

            # dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=output_dir)

            # Plots
            aa = pd.read_csv(f"{output_dir}/var_{name}.dat", header=None)
            iterations = []
            variable_values1 = []
            variable_values2 = []

            for index, row in aa.iterrows():
                parts = row[0].split()  # Split the string by whitespace
                iteration = int(parts[0])  # Extract the iteration as integer
                variable_value1 = float(parts[1][1:])  # Extract the variable value as float
                variable_value2 = float(row[1][1:-1])
                iterations.append(iteration)
                variable_values1.append(variable_value1)
                variable_values2.append(variable_value2)

            # Create a DataFrame with the extracted values
            aa_processed = pd.DataFrame({'Iteration': iterations, 'h': variable_values1, 'alpha': variable_values2})

            
            # # Plot the two variables versus the number of epochs
            # plt.plot(aa_processed['Iteration'], aa_processed['h'], "r-", label='h')
            # plt.plot(aa_processed['Iteration'], aa_processed['alpha'], "b-", label='alpha')
            # plt.xlabel("Iteration")
            # plt.ylabel("Variable Value")
            # plt.title(f"Variables vs Iteration, {name}")
            # plt.yscale('log')
            # plt.legend()
            # plt.yticks([0.1, 0.5, 1, 5, 10])
            # plt.grid(True)
            # plt.savefig(f'{output_dir}/var_{name}.png')
            # plt.show()

            log_file.write('{}: h = {}, alpha = {}\n'.format(name, variable_values1[-1], variable_values2[-1]))


            index = i * len(pp) + j
            axes[i, j].plot(aa_processed['Iteration'], aa_processed['h'], "r-", label='h')
            axes[i, j].plot(aa_processed['Iteration'], aa_processed['alpha'], "b-", label='alpha')
            axes[i, j].set_xlabel("Iteration")
            axes[i, j].set_ylabel("Variable Value")
            axes[i, j].set_title('{}: h = {}, alpha = {}'.format(name, variable_values1[-1], variable_values2[-1]))
            axes[i, j].set_yscale('log')
            axes[i, j].legend()
            axes[i, j].set_yticks([0.1, 0.5, 1, 5, 10])
            axes[i, j].grid(True)


plt.savefig(f'{script_directory}/identification{sub}/var_all_dates_phantoms.png')
plt.show()
