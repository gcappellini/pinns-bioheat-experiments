import os
import subprocess
from omegaconf import OmegaConf
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)



# cfg = OmegaConf.load(f"{src_dir}/config.yaml")

# exps = ["meas_1", "meas_2"]

# for exp in range(len(exps)):
#     cfg.experiment.name = ["cooling", exps[exp]]
#     OmegaConf.save(cfg, f"{src_dir}/config.yaml")
#     subprocess.run(["python", f'{src_dir}/main.py'])
#     subprocess.run(["python", f'{src_dir}/ic_compatibility_conditions.py'])



import numpy as np

def extract_matching(tot_true, tot_pred):
    # Extract the columns
    x_true, t_true, solution_true = tot_true[:, 0], tot_true[:, 1], tot_true[:, 2]
    x_pred, t_pred, solution_pred = tot_pred[:, 0], tot_pred[:, 1], tot_pred[:, 2]

    # Round the second column of tot_true to 2 decimal places
    t_true_rounded = np.round(t_true, 2)

    # Initialize an empty list to store the new array
    new_data = []

    # Loop through each row in tot_true and check if there is a match in tot_pred
    for i in range(tot_true.shape[0]):
        x_val = x_true[i]
        t_val = t_true_rounded[i]
        
        # Find indices in tot_pred that match both the first and second column values
        mask = (x_pred == x_val) & (np.round(t_pred, 2) == t_val)
        
        # If there is a match, add the row from tot_true and corresponding solution from tot_pred
        if np.any(mask):
            # Get the corresponding solution from tot_pred (first match found)
            pred_solution = solution_pred[mask][0]
            
            # Append tot_true's data along with the matched solution from tot_pred
            new_data.append([x_val, t_true[i], solution_true[i], pred_solution])

    # Convert new_data to a numpy array
    return np.array(new_data)