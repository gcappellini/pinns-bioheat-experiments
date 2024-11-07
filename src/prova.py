import utils as uu
import os
import numpy as np
from omegaconf import OmegaConf

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

conf = OmegaConf.load(f"{src_dir}/config.yaml")
x = np.linspace(0, 1, num=1000)

K = conf.model_properties.K

def new_ic_obs(x, b1, b2):
    # b1 = conf.model_properties.b1
    # b2 = conf.model_properties.b2
    return (1-x**b1)*(np.exp(-50/(x+0.001))+b2)

sistema = uu.ic_sys(x)
osservatore = uu.ic_obs(x)
new_oss = new_ic_obs(x, 1.4, 0.818)

theta_0 = sistema[0]
thetahat_0 = osservatore[0]
new_thetahat_0 = new_oss[0]

delta = x[1]-x[0]

theta_delta = sistema[1]
thetahat_delta = osservatore[1]
new_thetahat_delta = new_oss[1]

dtheta = (theta_delta - theta_0)/delta
dthetahat = (thetahat_delta - thetahat_0)/delta
new_dthetahat = (new_thetahat_delta - new_thetahat_0)/delta

print(theta_0, dtheta, new_thetahat_0, new_dthetahat)
print(dtheta - K *theta_0 , new_dthetahat - K*new_thetahat_0)



