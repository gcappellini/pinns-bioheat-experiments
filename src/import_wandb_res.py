import pandas as pd 
import wandb
import os

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
conf_dir = os.path.join(src_dir, "configs")
models = os.path.join(git_dir, "models")
tests_dir = os.path.join(git_dir, "tests")

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("ht-pinns/2025-02-23_opt_sp_direct")

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)
    # Keep only the specified keys in each summary dictionary
    keys_to_keep = ["L2RE", "MSE", "_runtime", "max", "mean", "test"]
    filtered_summary_list = [{k: v for k, v in summary.items() if k in keys_to_keep} for summary in summary_list]
    summary_list = filtered_summary_list

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)


    
# Flatten the summary and config dictionaries and merge them into a single dictionary for each run
flattened_data = []
for summary, config, name in zip(summary_list, config_list, name_list):
    flattened_dict = {**config, **summary}
    flattened_data.append(flattened_dict)

# Create a DataFrame from the flattened data
runs_df = pd.DataFrame(flattened_data)
# Order the DataFrame by the 'test' column from lower to greater
runs_df.rename(columns={'_runtime': 'runtime', 'test': 'loss test'}, inplace=True)
runs_df = runs_df.sort_values(by='loss test', ascending=True)
cols = runs_df.columns.tolist()
new_cols = ['num_domain', 'num_boundary', 'resampling', 'runtime', 'L2RE', 'MSE', 'max', 'mean', 'loss test']
runs_df = runs_df[new_cols]

out_dir = f"{tests_dir}/wandb_results"
os.makedirs(out_dir, exist_ok=True)
runs_df.to_csv(f"{out_dir}/sp_direct.csv", index=False)
