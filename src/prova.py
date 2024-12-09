import os
import numpy as np
from omegaconf import OmegaConf
import hydra
# import common as co
import plots as pp
import coeff_calc as cc
from scipy import integrate
import utils as uu
import time
import torch.nn as nn
import deepxde as dde
from common import set_run, generate_config_hash

np.random.seed(237)

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
conf_dir = os.path.join(src_dir, "configs")
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
models = os.path.join(git_dir, "models")
os.makedirs(tests_dir, exist_ok=True)

fold = f"{tests_dir}/transfer_learning"
os.makedirs(fold, exist_ok=True)

iters=2000

# Step 0: load model with n_ins=2
direct_conf_hash = "d77a6425b457ca35361d1fbdf12d452c"
conf = OmegaConf.load(f"{models}/config_{direct_conf_hash}.yaml")
model_2_ins_W0 = uu.check_for_trained_model(conf)

# Change perfusion and number of iterations
conf.model_properties.iters = 10
conf.model_properties.iters_lbfgs = 6955
conf.model_properties.W = 0.003303
conf.model_parameters.W_sys = 0.003303
conf.model_parameters.W_index = 7
conf.model_properties.optimizer = "adam"
model_2_ins_W7 = uu.compile_optimizer_and_losses(model_2_ins_W0, conf)
callbacks = uu.create_callbacks(conf)
hash_2_ins_W7 = generate_config_hash(conf.model_properties)
model_2_ins_W7_path = f"{fold}/model_{hash_2_ins_W7}"

losshistory, trainstate = model_2_ins_W7.train(
    iterations=iters,
    callbacks=callbacks,
    model_save_path=model_2_ins_W7_path,
    display_every=conf.plot.display_every
)

# train loss, test loss, plots












# # Step 1: augment input dimensionality to n_ins=3 to account for y2
# linears_2_ins = restored_model_2_ins.net.linears
# new_first_layer_3_ins = nn.Linear(in_features=3, out_features=50, bias=True)
# nn.init.xavier_uniform_(new_first_layer_3_ins.weight)
# linears_2_ins[0] = new_first_layer_3_ins

# # Step 2: create another model with net as before and new data, and train with adam
# conf.model_properties.n_ins=3
# data_3_ins = uu.create_model(conf)
# model_3_ins = dde.Model(data=data_3_ins.data, net=restored_model_2_ins.net)
# conf.model_properties.optimizer = "adam"
# model_3_ins = uu.compile_optimizer_and_losses(model_3_ins, conf)
# callbacks = uu.create_callbacks(conf)

# losshistory, trainstate = model_3_ins.train(
#     iterations=iters,
#     callbacks=callbacks,
#     model_save_path=f"{fold}/model_3_ins_adam",
#     display_every=conf.plot.display_every
# )

# # Step 3: train with LBFGS
# conf.model_properties.optimizer = "L-BFGS"
# model_3_ins = uu.compile_optimizer_and_losses(model_3_ins, conf)
# callbacks = uu.create_callbacks(conf)

# losshistory, trainstate = model_3_ins.train(
#     iterations=iters,
#     callbacks=callbacks,
#     model_save_path=f"{fold}/model_3_ins_L-BFGS",
#     display_every=conf.plot.display_every
# )

# # Step 4: augment input dimensionality to n_ins=4 to account for y1
# linears_3_ins = model_3_ins.net.linears
# new_first_layer_4_ins = nn.Linear(in_features=4, out_features=50, bias=True)
# nn.init.xavier_uniform_(new_first_layer_4_ins.weight)
# linears_3_ins[0] = new_first_layer_4_ins

# # Step 5: create another model with net as before and new data, and train with adam
# conf.model_properties.n_ins=4
# data_4_ins = uu.create_model(conf)
# model_4_ins = dde.Model(data=data_4_ins.data, net=model_3_ins.net)
# conf.model_properties.optimizer = "adam"
# model_4_ins = uu.compile_optimizer_and_losses(model_4_ins, conf)
# callbacks = uu.create_callbacks(conf)

# losshistory, trainstate = model_4_ins.train(
#     iterations=iters,
#     callbacks=callbacks,
#     model_save_path=f"{fold}/model_4_ins_adam",
#     display_every=conf.plot.display_every
# )

# # Step 6: train with LBFGS
# conf.model_properties.optimizer = "L-BFGS"
# model_4_ins = uu.compile_optimizer_and_losses(model_4_ins, conf)
# callbacks = uu.create_callbacks(conf)

# losshistory, trainstate = model_4_ins.train(
#     iterations=iters,
#     callbacks=callbacks,
#     model_save_path=f"{fold}/model_4_ins_L-BFGS",
#     display_every=conf.plot.display_every
# )

# # Step 7: augment input dimensionality to n_ins=5 to account for y3
# linears_4_ins = model_4_ins.net.linears
# new_first_layer_5_ins = nn.Linear(in_features=5, out_features=50, bias=True)
# nn.init.xavier_uniform_(new_first_layer_5_ins.weight)
# linears_4_ins[0] = new_first_layer_5_ins

# # Step 8: create another model with net as before and new data, and train with adam
# conf.model_properties.n_ins=5
# data_5_ins = uu.create_model(conf)
# model_5_ins = dde.Model(data=data_5_ins.data, net=model_4_ins.net)
# conf.model_properties.optimizer = "adam"
# model_5_ins = uu.compile_optimizer_and_losses(model_5_ins, conf)
# callbacks = uu.create_callbacks(conf)

# losshistory, trainstate = model_5_ins.train(
#     iterations=iters,
#     callbacks=callbacks,
#     model_save_path=f"{fold}/model_5_ins_adam",
#     display_every=conf.plot.display_every
# )

# # Step 9: train with LBFGS
# conf.model_properties.optimizer = "L-BFGS"
# model_5_ins = uu.compile_optimizer_and_losses(model_5_ins, conf)
# callbacks = uu.create_callbacks(conf)

# losshistory, trainstate = model_5_ins.train(
#     iterations=iters,
#     callbacks=callbacks,
#     model_save_path=f"{fold}/model_5_ins_L-BFGS",
#     display_every=conf.plot.display_every
# )

