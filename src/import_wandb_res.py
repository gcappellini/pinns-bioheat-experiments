import pandas as pd 
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("ht-pinns/2025-02-23_opt_sp_direct")

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

    # Keep only the specified keys in each summary dictionary
    keys_to_keep = ["L2RE", "MSE", "_runtime", "_step", "max", "mean", "test"]
    filtered_summary_list = [{k: v for k, v in summary.items() if k in keys_to_keep} for summary in summary_list]

    summary_list = filtered_summary_list
    
runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("project.csv")