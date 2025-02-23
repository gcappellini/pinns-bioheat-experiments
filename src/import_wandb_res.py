import pandas as pd 
import wandb
import os
import yaml
import csv

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
conf_dir = os.path.join(src_dir, "configs")
models = os.path.join(git_dir, "models")
tests_dir = os.path.join(git_dir, "tests")

# api = wandb.Api()

# # Project is specified by <entity/project-name>
# runs = api.runs("ht-pinns/2025-02-23_opt_sp_direct_1")

# summary_list, config_list, name_list = [], [], []
# for run in runs: 
#     # .summary contains the output keys/values for metrics like accuracy.
#     #  We call ._json_dict to omit large files 
#     summary_list.append(run.summary._json_dict)
#     # Keep only the specified keys in each summary dictionary
#     keys_to_keep = ["L2RE", "MSE", "runtime", "max", "mean", "testloss"]
#     filtered_summary_list = [{k: v for k, v in summary.items() if k in keys_to_keep} for summary in summary_list]
#     summary_list = filtered_summary_list

#     # .config contains the hyperparameters.
#     #  We remove special values that start with _.
#     config_list.append(
#         {k: v for k,v in run.config.items()
#           if not k.startswith('_')})

#     # .name is the human-readable name of the run.
#     name_list.append(run.name)


    
# # Flatten the summary and config dictionaries and merge them into a single dictionary for each run
# flattened_data = []
# for summary, config, name in zip(summary_list, config_list, name_list):
#     flattened_dict = {**config, **summary}
#     flattened_data.append(flattened_dict)

# # Create a DataFrame from the flattened data
# runs_df = pd.DataFrame(flattened_data)
# # Order the DataFrame by the 'test' column from lower to greater
# runs_df.rename(columns={'MSE':'mse','num_domain':'nres', 'num_boundary':'nb'}, inplace=True)
# cols = runs_df.columns.tolist()
# new_cols = ['nres','nb','resampling','runtime','testloss','mse']
# runs_df = runs_df[new_cols]

# pd.options.display.float_format = '{:.1e}'.format
# runs_df["runtime"] = runs_df["runtime"].apply(lambda x: f"{x:.1f}")
# runs_df = runs_df.sort_values(by='testloss', ascending=True)
out_dir = f"{tests_dir}/wandb_results"
# os.makedirs(out_dir, exist_ok=True)
# runs_df.to_csv(f"{out_dir}/sp_direct_1.csv", index=False)



# # Load YAML file
# with open(f"{conf_dir}/pinn_hp/config_direct.yaml", "r") as file:  # Change filename if needed
#     data = yaml.safe_load(file)
#     # Rename keys in data
#     key_mapping = {
#         "activation": "af",
#         "initial_weights_regularizer": "iwr",
#         "initialization": "init",
#         "learning_rate": "lr",
#         "num_dense_layers": "depth",
#         "num_dense_nodes": "width",
#         "num_domain": "nres",
#         "num_boundary": "nb",
#         "num_test": "ntest",
#         "w_res": "wres",
#         "w_bc0": "wbc0",
#         "n_ins": "nins",
#         "n_anchor_points": "nanc"
#     }

#     # Remove underscores from other entries
#     data = {key_mapping.get(k, k.replace('_', '')): v for k, v in data.items()}

# Load YAML file
with open(f"{conf_dir}/model_properties/simulation.yaml", "r") as file:  # Change filename if needed
    data = yaml.safe_load(file)
    # Rename keys in data
    key_mapping = {
        "tauf": "tf",
        "Troom": "Tf",
        "pwr_fact": "p",
        "PD": "pd",
        "W_sys": "wb",
        "Tgt20": "Tgt0",
    }

    # Remove underscores from other entries
    data = {key_mapping.get(k, k.replace('_', '')): v for k, v in data.items()}
# Write to CSV file
csv_filename = f"{out_dir}/props_direct.csv"
with open(csv_filename, "w", newline="") as file:
    writer = csv.writer(file)
    for key, value in data.items():
        writer.writerow([key, value])






