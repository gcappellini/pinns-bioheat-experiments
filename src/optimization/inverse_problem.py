import utils_meas as utils
import os
import deepxde as dde
import numpy as np
from deepxde.backend import torch
from scipy.interpolate import interp1d


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)



C1 = dde.Variable(67.3)
C2 = dde.Variable(150.0)


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    return (
        C1 * dy_t - dy_xx + C2 * y
    )

def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

a = utils.import_testdata(3)


locations = np.unique(a[:, 0])
rows_t0 = a[a[:, 1] == 0.0]
t0 = rows_t0[:, -2].reshape(len(locations),)
func_ic = interp1d(locations, t0, kind='previous')

instants = np.unique(a[:, 1])

rows_x1 = a[a[:, 0] == 1.0]
rows_x0 = a[a[:, 0] == 0.0]

x0 = rows_x0[:, -2].reshape(len(instants),)
func_bc0 = interp1d(instants, x0, kind='previous')

x1 = rows_x1[:, -2].reshape(len(instants),)
func_bc1 = interp1d(instants, x1, kind='previous')

def func(z):

    list = []
    for el in z:
        if el[0]==0.0:
            list.append(func_bc0(el[1]))
        if el[0]==1.0:
            list.append(func_bc1(el[1]))
        if el[1]==0.0:
            list.append(func_ic(el[0]))
        # else:
        #     print("Error")
    out = np.array(list)
    return out


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc0 = dde.icbc.DirichletBC(geomtime, func, boundary_0)
bc1 = dde.icbc.DirichletBC(geomtime, func, boundary_1)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

observe_x = a[:, 0:2]
observe_y = dde.icbc.PointSetBC(observe_x, a[:, -2].reshape(len(observe_x), 1), component=0)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc0, bc1, ic, observe_y],
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    anchors=observe_x,
    # solution=func,
    num_test=10000,
)

layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile(
    # "adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=C
    "adam", lr=0.001, external_trainable_variables=[C1, C2]
)
variable1 = dde.callbacks.VariableValue(C1, period=1000)
variable2 = dde.callbacks.VariableValue(C2, period=1000)
losshistory, train_state = model.train(iterations=50000, callbacks=[variable1, variable2])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)