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
from simulation import load_ground_truth

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

# Step 0: load model with n_ins=2
conf = OmegaConf.load(f"{conf_dir}/config_run.yaml")
model_no_rff = uu.check_for_trained_model(conf)


# Step 1: augment input dimensionality to n_ins=3 to account for y2
linears_2_ins = model_no_rff.net.linears
print(model_no_rff.net)
print(linears_2_ins)

# new_rff_layer = nn.Linear(in_features=3, out_features=50, bias=True)
# nn.init.xavier_uniform_(new_first_layer_3_ins.weight)
# linears_2_ins[0] = new_first_layer_3_ins












# Change perfusion and number of iterations
# conf.model_properties.iters = 2000
# conf.model_properties.iters_lbfgs = 6955
# conf.model_properties.W = 0.001506
# conf.model_parameters.W_sys = 0.001506
# conf.model_parameters.W_index = 1
# conf.model_properties.optimizer = "adam"
# model_2_ins_W1 = uu.compile_optimizer_and_losses(model_2_ins_W0, conf)
# callbacks = uu.create_callbacks(conf)

# losshistory, trainstate = model_2_ins_W1.train(
#     iterations=conf.model_properties.iters,
#     callbacks=callbacks,
#     # model_save_path=model_2_ins_W7_path,
#     display_every=conf.plot.display_every
# )

# conf.model_properties.optimizer = "L-BFGS"
# hash_2_ins_W1 = generate_config_hash(conf.model_properties)
# model_2_ins_W1_path = f"{fold}/model_{hash_2_ins_W1}"
# # dde.config.set_default_float("float64")
# dde.optimizers.config.set_LBFGS_options(maxcor=100, 
#                                         ftol=1e-08, 
#                                         gtol=1e-08, 
#                                         maxiter=conf.model_properties.iters_lbfgs, 
#                                         maxfun=None, maxls=50)

# model_2_ins_W1 = uu.compile_optimizer_and_losses(model_2_ins_W1, conf)
# callbacks = uu.create_callbacks(conf)

# losshistory_lbfgs_2_ins_W1, trainstate = model_2_ins_W1.train(
#     iterations=conf.model_properties.iters_lbfgs,
#     callbacks=callbacks,
#     model_save_path=model_2_ins_W1_path,
#     display_every=conf.plot.display_every
# )

# final_train, final_test = losshistory_lbfgs_2_ins_W1.loss_train, losshistory_lbfgs_2_ins_W1.loss_test
# train, test = np.array(final_train).sum(axis=1).ravel(), np.array(final_test).sum(axis=1).ravel()
# print(train[-1], test[-1])

# system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(conf, f"{tests_dir}/cooling_simulation/ground_truth")
# system = uu.get_pred(model_2_ins_W1, system_gt["grid"], fold, "system")
# uu.check_and_wandb_upload(
#     system_gt=system_gt,
#     system=system,
#     conf=conf,
#     output_dir=fold
# )