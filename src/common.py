import deepxde as dde
import numpy as np
import os
import torch
import json
import hashlib
import logging
from omegaconf import OmegaConf

dde.config.set_random_seed(200)

# device = torch.device("cpu")
device = torch.device("cuda")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)

logger = logging.getLogger(__name__)

def setup_log(string):
    # logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info(string)
    # return logger

models = os.path.join(git_dir, "models")
os.makedirs(models, exist_ok=True)

run_figs = [None]


def set_run(prj_figs, cfg, run):
    global run_figs

    # run_figs = os.path.join(prj_figs, run)
    # os.makedirs(run_figs, exist_ok=True)

    props = cfg.model_properties
    pars = cfg.model_parameters
    simu_settings = getattr(cfg.experiment_type, "simulation")

    if run.startswith("ground_truth"):
        
        cfg.output_dir = os.path.abspath(prj_figs)
        # cfg.output_dir = os.path.abspath(os.path.join(prj_figs, run))
        os.makedirs(cfg.output_dir, exist_ok=True)
        props.Ty10, props.Ty20, props.Ty30, props.Tgt20 = simu_settings.Ty10, simu_settings.Ty20, simu_settings.Ty30, simu_settings.Tgt20
        pars.lam, pars.upsilon = simu_settings.lam, simu_settings.upsilon
        OmegaConf.save(cfg, f"{conf_dir}/config_run.yaml")
        cfg = filter_config_for_matlab(cfg)
        run_figs = cfg.output_dir

    elif run == "simulation_system":
        props.W = pars.W_sys
        props.Ty10, props.Ty20, props.Ty30 = simu_settings.Ty10, simu_settings.Ty20, simu_settings.Ty30
        pars.lam, pars.upsilon = simu_settings.lam, simu_settings.upsilon
        props.n_ins = 2

    elif run.startswith("simulation"):
        
        props.Ty10, props.Ty20, props.Ty30, props.Tgt20 = simu_settings.Ty10, simu_settings.Ty20, simu_settings.Ty30, simu_settings.Tgt20
        pars.lam, pars.upsilon = simu_settings.lam, simu_settings.upsilon
        if run == "simulation_mm_obs":
            cfg.output_dir = prj_figs

    elif run.startswith("meas_cool"):
        props.h, props.pwr_fact = 10.0, 0.0
        meas_settings = getattr(cfg.experiment_type, run)
        props.Ty10, props.Ty20, props.Ty30 = meas_settings.Ty10, meas_settings.Ty20, meas_settings.Ty30
        pars.lam, pars.upsilon = meas_settings.lam, meas_settings.upsilon


    elif run.startswith("hpo"):
        pars.n_obs = 1
        props.n_ins = 4
        cfg.experiment.ground_truth = False
        run_figs = os.path.join(prj_figs, run)
        os.makedirs(run_figs, exist_ok=True)
        cfg.output_dir = run_figs
    
    elif run.startswith("inverse"):
        pars.n_obs = 0
        props.n_ins = 2
        props.W = pars.W7


    OmegaConf.save(cfg, f"{prj_figs}/config_{run}.yaml")
    # OmegaConf.save(cfg, f"{conf_dir}/config_{run}.yaml")

    return run_figs, cfg


def filter_config_for_matlab(cfg):
    cfg_matlab = OmegaConf.create({
        "model_properties": cfg.model_properties,
        "model_parameters": cfg.model_parameters,
        "output_dir": cfg.output_dir,
        "experiment": cfg.experiment.run,
        })
    OmegaConf.save(cfg_matlab, f"{conf_dir}/config_ground_truth.yaml")
    return cfg_matlab

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

# if __name__ == "__main__":
