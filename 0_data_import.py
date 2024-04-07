import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)

nn = ["240124", "240125", "240125b", "240130", "240202"]
pp = ["P1", "P2"]

output_dir = f"{script_directory}/measurements"

os.makedirs(output_dir, exist_ok=True)


for date in nn:
    for phantom in pp:

        name = f"{date}_{phantom}"

        # Read from a specific sheet
        df = pd.read_excel(f'{script_directory}/measurements/{date}.xlsx', sheet_name=f'{phantom}', header=0, nrows=235)
        # Scaling time
        first_date_value = df['date'].iloc[0]

        # Ensure 'date' is a pandas Timestamp object
        df['date'] = pd.to_datetime(df['date'])

        df['time'] = ((df['date'] - first_date_value).dt.total_seconds() / 60).round().astype(int)
        df = df.drop('date', axis=1)
        df['time'] = df['time']/len(df['time'])

        # Scaling temperature
        max_T = 36.7
        min_T = 22.0
        df_transformed = (df.drop('time', axis=1) - min_T) / (max_T - min_T)

        # df_transformed = df.drop('time', axis=1)
        # Combine the transformed DataFrame with the 'time' column
        df_result = pd.concat([df['time'], df_transformed], axis=1)

        # Create a new DataFrame to store the transformed values
        dfs = []

        # Iterate over unique values of 'time' in df_result
        for time_value in df_result['time'].unique():
            # Generate 7 evenly spaced values between 0 and 1 for 'position'
            positions = np.linspace(0, 1, 7)

            # Extract 'theta' values for the current 'time' from df_result
            theta_values = df_result[df_result['time'] == time_value][
                ['P6', 'P5', 'P4', 'P3', 'P2', 'P1', 'P0']].values.flatten()

            # Create a DataFrame for the current 'time' with the specified columns and values
            time_df = pd.DataFrame({
                'position': positions,
                'time': [time_value] * 7,
                'theta': theta_values
            })

            # Append the current 'time' DataFrame to the list
            dfs.append(time_df)

        # Concatenate all DataFrames in the list into a single DataFrame
        new_df = pd.concat(dfs, ignore_index=True)

        # print(new_df)

        np.savez(f"{output_dir}/meas_{name}.npz", x=new_df['position'], t=new_df['time'], theta=new_df['theta'])

        # Create a new DataFrame to store the transformed values
        obs_dfs = []

        # Iterate over unique values of 'time' in df_result
        for time_value in df_result['time'].unique():
            # Generate 7 evenly spaced values between 0 and 1 for 'position'
            positions = np.linspace(0, 1, 7)

            # Extract 'theta' values for the current 'time' from df_result
            t_0_values = np.full_like(positions, df_result[df_result['time'] == time_value]['P6'])
            t_1_values = np.full_like(positions, df_result[df_result['time'] == time_value]['P0'])
            t_bolus_values = np.full_like(positions, df_result[df_result['time'] == time_value]['bolus'])

            # Create a DataFrame for the current 'time' with the specified columns and values
            obs_df = pd.DataFrame({
                'position': positions,
                'time': [time_value] * 7,
                # 'theta': theta_values,
                't_0': t_0_values,
                't_1': t_1_values,
                't_bolus': t_bolus_values,
            })

            # Append the current 'time' DataFrame to the list
            obs_dfs.append(obs_df)

        # Concatenate all DataFrames in the list into a single DataFrame
        observing = pd.concat(obs_dfs, ignore_index=True)
        # print(observing)
        np.savez(f"{output_dir}/obs_{name}.npz", x=observing['position'], t=observing['time'],
                 t_0=observing['t_0'], t_1=observing['t_1'], t_bolus=observing['t_bolus'])

# Initialize a figure with subplots based on the number of elements in nn and pp
fig, axs = plt.subplots(len(pp), len(nn), figsize=(13, 7))

for i, date in enumerate(nn):
    for j, phantom in enumerate(pp):
        # Load the saved data
        data = np.load(f"{output_dir}/meas_{date}_{phantom}.npz")
        x = data['x']
        t = data['t']
        theta = data['theta']

        la = len(np.unique(x))
        le = len(np.unique(t))

        d = theta.max() if theta.max() > 1 else 1
        # Plot theta vs x and t using imshow
        im = axs[j, i].imshow(theta.reshape((le, la)), aspect='auto', origin='lower', 
                              extent=[np.unique(x).min(), np.unique(x).max(), np.unique(t).min(), 1], cmap='inferno', vmin=0, vmax=d)
        axs[j, i].set_title(f"{date}_{phantom}")
        axs[j, i].set_xlabel('z')
        axs[j, i].set_ylabel('t')
        plt.colorbar(im, ax=axs[j, i])

# Adjust layout
plt.tight_layout()
plt.savefig(f'{output_dir}/all_measurements.png')
plt.show()


# Initialize a figure with subplots based on the number of elements in nn and pp


for j, phantom in enumerate(pp):
    fig, axs = plt.subplots(len(nn), 3, figsize=(10, 9))
    fig.subplots_adjust(hspace=1.5) 
    for i, date in enumerate(nn):
        # Load the saved data
        obs = np.load(f"{output_dir}/obs_{date}_{phantom}.npz")
        t, t_0, t_1, t_bolus = obs["t"], obs["t_0"], obs["t_1"], obs["t_bolus"] 

        # Plot observing['t_inf'] vs observing['time'] on the left subplot
        axs[i, 0].plot(t, t_0, 'r-', label='t_inf')
        axs[i, 0].set_xlabel('time')
        axs[i, 0].set_title(r'$\tilde y_1(\tau)$')
        # axs[0].set_title('t_0 vs Time')

        # Plot observing['t_sup'] vs observing['time'] on the center subplot
        axs[i, 1].plot(t, t_1, 'g-', label='t_sup')
        axs[i, 1].set_xlabel('time')
        # axs[1].set_ylabel('t_1')
        axs[i, 1].set_title(r'$\tilde y_2(\tau)$')

        # Plot observing['t_bolus'] vs observing['time'] on the right subplot
        axs[i, 2].plot(t, t_bolus, 'b-', label='t_bolus')
        axs[i, 2].set_xlabel('time')
        # axs[2].set_ylabel(r'$t_bolus$')
        axs[i, 2].set_title(r'$\tilde y_3(\tau)$')

        axs[i, 0].text(0.0, 1.3, f'{date}', horizontalalignment='center', verticalalignment='center', transform=axs[i, 0].transAxes, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{output_dir}/obs_{phantom}.png')
    plt.show()


