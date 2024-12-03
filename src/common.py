import deepxde as dde
import numpy as np
import os
# import matplotlib.pyplot as plt
import torch
# import seaborn as sns
# import wandb
import json
# from scipy.interpolate import interp1d
# from scipy import integrate
# import pickle
# import pandas as pd
import hashlib
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


models = os.path.join(git_dir, "models")
os.makedirs(models, exist_ok=True)

run_figs = [None]


def set_run(prj_figs, cfg, run):
    global run_figs

    run_figs = os.path.join(prj_figs, run)
    os.makedirs(run_figs, exist_ok=True)

    if run=="ground_truth":
        cfg = filter_config_for_matlab(cfg)
    
    if run=="simulation_system":
        cfg.model_properties.W = cfg.model_parameters.W_sys
        cfg.model_properties.n_ins = 2
        cfg.model_properties.b1, cfg.model_properties.b2, cfg.model_properties.b3 = None, None, None

    if run=="simulation_mm_obs":
        cfg.model_properties.n_ins = 3

    OmegaConf.save(cfg, f"{run_figs}/config.yaml")
    OmegaConf.save(cfg, f"{conf_dir}/config_{run}.yaml")

    return run_figs, cfg


def filter_config_for_matlab(cfg):
    cfg_matlab = OmegaConf.create({
        "model_properties": cfg.model_properties,
        "model_parameters": cfg.model_parameters,
        "output_dir": cfg.output_dir
        })
    
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