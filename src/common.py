import deepxde as dde
import numpy as np
import os
import torch
import json
import hashlib

from omegaconf import OmegaConf


# device = torch.device("cpu")
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)



# def setup_log(string):
#     # logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
#     #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
#     logger.info(string)
    # return logger

models = os.path.join(git_dir, "models")
os.makedirs(models, exist_ok=True)

run_figs = [None]

def set_seed(seed):
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_run(prj_figs, cfg, run):
    global run_figs

    experiments_cfg = OmegaConf.load(f"{conf_dir}/experiments.yaml")
    
    props = cfg.properties
    pars = cfg.parameters
    temps = cfg.temps
    pdecoeff = cfg.pdecoeff
    hp = cfg.hp
    set_seed(cfg.hp.seed)

    # if run.startswith("ground_truth"):   
    #     simu_settings = getattr(experiments_cfg, "simulation")
    #     cfg.output_dir = os.path.abspath(prj_figs)
    #     # cfg.output_dir = os.path.abspath(os.path.join(prj_figs, run))
    #     os.makedirs(cfg.output_dir, exist_ok=True)
    #     temps.Ty10, temps.Ty20, temps.Ty30, temps.Tgt0 = simu_settings.Ty10, simu_settings.Ty20, simu_settings.Ty30, simu_settings.Tgt20
    #     pars.lam, pars.upsilon = simu_settings.lam, simu_settings.upsilon
    #     OmegaConf.save(cfg, f"{conf_dir}/config_run.yaml")
    #     cfg_matlab = filter_config_for_matlab(cfg)
    #     run_figs = cfg_matlab.output_dir

    if run == "simulation_system":
        simu_settings = getattr(experiments_cfg, "simulation")
        pars.nobs=0
        pdecoeff.wb = pars.wbsys
        temps.Ty10, temps.Ty20, temps.Ty30 = simu_settings.Ty10, simu_settings.Ty20, simu_settings.Ty30
        pars.ag, pars.ups = simu_settings.lam, simu_settings.upsilon
        hp.nins = 2


    elif run.startswith("simulation"):
        simu_settings = getattr(experiments_cfg, "simulation")
        temps.Ty10, temps.Ty20, temps.Ty30, temps.Tgt0 = simu_settings.Ty10, simu_settings.Ty20, simu_settings.Ty30, simu_settings.Tgt20
        pars.ag, pars.ups = simu_settings.lam, simu_settings.upsilon
        if np.isin(run, ["simulation_mm_obs", "simulation_ground_truth"]):
            cfg.output_dir = os.path.abspath(prj_figs)
            os.makedirs(cfg.output_dir, exist_ok=True)
        elif run == "simulation_system":
            pdecoeff.wb = pars.wbsys
            hp.nins = 2


    elif run.startswith("meas_cool"):
        simu_settings = getattr(experiments_cfg, run)
        props.h, props.pwrfact = 10.0, 0.0
        meas_settings = getattr(experiments_cfg, run)
        temps.Ty10, temps.Ty20, temps.Ty30 = meas_settings.Ty10, meas_settings.Ty20, meas_settings.Ty30
        pars.ag, pars.ups = meas_settings.lam, meas_settings.upsilon

    elif run.startswith("hpo"):
        pars.nobs = 1
        hp.nins = 4
        cfg.experiment.ground_truth = False
        run_figs = os.path.join(prj_figs, run)
        os.makedirs(run_figs, exist_ok=True)
        cfg.output_dir = run_figs
    

    elif run.startswith("inverse"):
        pars.nobs = 0
        hp.nins = 2
        # pdecoeff.W = pars.W7

    cfg = calc_coeff(cfg)
    # if run=="simulation_ground_truth":
    #     filter_config_for_matlab(cfg)
    # OmegaConf.save(cfg, f"{prj_figs}/config_{run}.yaml")
    # OmegaConf.save(cfg, f"{conf_dir}/config_{run}.yaml")

    return run_figs, cfg



def generate_config_hash(config_data):

    # Convert OmegaConf object to a dictionary (nested structure)
    config_dict = OmegaConf.to_container(config_data, resolve=True)
    
    # Convert the dictionary to a sorted JSON string
    config_string = json.dumps(config_dict, sort_keys=True)  # Sort to ensure consistent ordering
    
    # Create a unique hash using MD5
    config_hash = hashlib.md5(config_string.encode()).hexdigest()
    
    return config_hash


def write_json(data, filepath):
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_data = {k: convert_to_serializable(v) for k, v in data.items()}

    with open(filepath, 'w') as file:
        json.dump(serializable_data, file, indent=4)


def find_matching_json(folder_path, target_dict):
    # Loop through all the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .json file
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                try:
                    # Load the content of the file
                    json_data = json.load(file)
                    
                    # Compare the content of the JSON file with the target dictionary
                    if json_data == target_dict:
                        print(f"Matching file found: {filename}")
                        return file_path
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON file: {filename}")
    
    # If no matching file is found
    print("No matching file found.")
    return None


def calculate_aicoeff(cfg):
    props = cfg.properties
    pdecoeff = cfg.pdecoeff
    dT = cfg.temps.Tmax - cfg.temps.Troom
    a1: float = round((props.L0**2/props.tf)*((props.rho*props.c)/props.k), 7)
    a2: float = round(props.L0**2*props.rhob*props.cb/props.k, 7)
    cc: float = round(np.log(2)/(props.PD - 10**(-2)*props.x0), 7)
    a3: float = round(props.pwrfact*props.rho*props.L0**2*props.beta*props.SAR0*np.exp(cc*props.x0)/props.k*dT, 7)
    a4: float = round(cc*props.L0, 7)
    a5: float = round(props.L0*props.h/props.k, 7)
    pdecoeff.a1 = float(a1)
    pdecoeff.a2 = float(a2)
    pdecoeff.a3 = float(a3)
    pdecoeff.a4 = float(a4)
    pdecoeff.a5 = float(a5)
    return cfg


def calculate_temps(cfg):
    temps = cfg.temps
    pdecoeff = cfg.pdecoeff
    Troom = temps.Troom
    Tmax = temps.Tmax

    def scale_t(t: float) -> float:
        return float(round((t - Troom) / (Tmax - Troom), 5))

    labels_theta = ['y10', 'y20', 'y30', 'thetagt0', 'thetagt10']
    scaled_temps = {key: scale_t(getattr(temps, key)) for key in ['Ty10', 'Ty20', 'Ty30', 'Tgt0', 'Tgt10']}
    for key, value in scaled_temps.items():
        setattr(pdecoeff, labels_theta[['Ty10', 'Ty20', 'Ty30', 'Tgt0', 'Tgt10'].index(key)], value)
    return cfg


def calculate_cicoeff(cfg):
    props = cfg.properties
    pdecoeff = cfg.pdecoeff
    hp = cfg.hp
    if hp.nins == 2:
        pdecoeff.c1, pdecoeff.c2, pdecoeff.c3 = None, None, None
    elif hp.nins > 2:
        pdecoeff.c3 = float(round(pdecoeff.y20, 5))
        pdecoeff.c2 = float(round(-pdecoeff.a5 * (pdecoeff.y30 - pdecoeff.y20), 5))
        pdecoeff.c1 = float(round(pdecoeff.y10 - pdecoeff.c2 - pdecoeff.c3, 5))
    return cfg


def calculate_bicoeff(cfg):
    pars = cfg.parameters
    pdecoeff = cfg.pdecoeff

    # Define the equations in matrix form
    A = np.array([
        [1, 1, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [pars.Xgt**3, pars.Xgt**2, pars.Xgt, 1]
    ])

    B = np.array([pdecoeff.y10, pdecoeff.y20, -pdecoeff.a5 * (pdecoeff.y30 - pdecoeff.y20), pdecoeff.thetagt0])
    sol = np.linalg.solve(A, B)
    pdecoeff.b1, pdecoeff.b2, pdecoeff.b3, pdecoeff.b4 = [float(round(val, 5)) for val in sol]
    return cfg


def calculate_pars(cfg):
    pars = cfg.parameters
    conv = cfg.convergence
    wbmin: float = pars.wbmin
    wbmax: float = pars.wbmax
    obsindex: int = pars.obsindex
    nobs: int = pars.nobs

    obs_steps = 8 if nobs<=8 else nobs
    obs = np.logspace(np.log10(wbmin), np.log10(wbmax), obs_steps).round(6)
    wbobs = float(obs[obsindex])
    pars.wbobs = wbobs

    eight_obs = np.logspace(np.log10(wbmin), np.log10(wbmax), 8).round(6)
    for i in range(8):
        setattr(pars, f'wb{i}', float(eight_obs[i]))

    if nobs ==1:
        cfg = calculate_conv_pars(cfg)
    else:
        conv.drdiff, conv.drexact, conv.c0 = None, None, None
    return cfg


def calculate_conv_pars(cfg):
    pdecoeff = cfg.pdecoeff
    pars = cfg.parameters
    props = cfg.properties
    conv = cfg.convergence
    pwic: float = np.where(pdecoeff.oig>=(np.pi**2)/4, (np.pi**2)/4, pdecoeff.oig)

    drexact: float = 2*(pwic/pdecoeff.a1+pars.wbsys*props.tf)
    c0: float = (np.abs(props.tf*(pars.wbobs - pars.wbsys)**2)/(pwic+ pars.wbobs*props.tf)**2)
    drdiff: float = (pwic/pdecoeff.a1+pars.wbobs*props.tf)/2
    conv.drdiff = float(round(drdiff, 7))
    conv.drexact = float(round(drexact, 7))
    conv.c0 = float(round(c0, 7))
    return cfg


def calc_coeff(cfg):
    cfg = calculate_aicoeff(cfg)
    cfg = calculate_temps(cfg)
    cfg = calculate_cicoeff(cfg)
    cfg = calculate_bicoeff(cfg)
    cfg = calculate_pars(cfg)
    cfg = calculate_conv_pars(cfg)
    return cfg

if __name__ == "__main__":
    cfg = OmegaConf.load(f"{conf_dir}/config_run.yaml")
    perfs = [6.3e-5, 1.207e-3, 1.651e-3, 3.303e-3]
    for pp in perfs:
        cfg.parameters.wbsys = 3.303e-3
        cfg.parameters.wbobs = pp
        # print("a1:", cfg.pdecoeff.a1)
        # print("oig:", cfg.pdecoeff.oig)
        # print("tf:", cfg.properties.tf)
        print("wbsys:", cfg.parameters.wbsys)
        print("wbobs:", cfg.parameters.wbobs)
        cfg = calculate_conv_pars(cfg)
        # print("drexact:", cfg.convergence.drexact)
        # print("drdiff:", cfg.convergence.drdiff)
        print("bounding error:", 4*cfg.convergence.c0)
        print("-------")