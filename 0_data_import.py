import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)

nn = ["240124", "240125", "240125b", "240130", "240202"]
pp = ["P1", "P2"]

for name in nn:
    for phantom in pp:

        output_dir = f"{script_directory}/measurements/{name}_{phantom}"

        os.makedirs(output_dir, exist_ok=True)

        # Read from a specific sheet
        df = pd.read_excel(f'{script_directory}/measurements/{name}.xlsx', sheet_name=f'{phantom}', header=0, nrows=270)
        # Scaling time
        first_date_value = df['date'].iloc[0]

        # Ensure 'date' is a pandas Timestamp object
        df['date'] = pd.to_datetime(df['date'])

        df['time'] = ((df['date'] - first_date_value).dt.total_seconds() / 60).round().astype(int)
        df = df.drop('date', axis=1)
        df['time'] = df['time']/len(df['time'])

        # Scaling temperature
        max_T = 36.05
        min_T = 22.0
        df_transformed = (df.drop('time', axis=1) - min_T) / (max_T - min_T)

        # Combine the transformed DataFrame with the 'time' column
        df_result = pd.concat([df['time'], df_transformed], axis=1)
        # print(df_result)

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

        np.savez(f"{output_dir}/meas_{name}_{phantom}.npz", x=new_df['position'], t=new_df['time'], theta=new_df['theta'])

        x = new_df['position'].values.reshape(-1, 1)
        y = new_df['time'].values.reshape(-1, 1)
        z = new_df['theta'].values.reshape(-1, 1)

        # Create a 2D plot using imshow
        plt.figure(figsize=(8, 6))
        # plt.imshow(z, extent=[x.min(), x.max(), y.min(), y.max()], cmap='inferno', aspect='auto', origin='lower', vmin=0, vmax=1)
        plt.scatter(x[::5], y[::5], s=20, c=z[::5], cmap="inferno", vmin=0, vmax=1)
        plt.colorbar(label="theta")

        # Set labels and title
        plt.xlabel('Position')
        plt.ylabel('Time')
        plt.title('2D Plot')

        # Save the plot
        plt.savefig(f'{output_dir}/2d_plot_{phantom}.png')

        # Show the plot
        plt.show()
        # Create a new DataFrame to store the transformed values
        obs_dfs = []

        # Iterate over unique values of 'time' in df_result
        for time_value in df_result['time'].unique():
            # Generate 7 evenly spaced values between 0 and 1 for 'position'
            positions = np.linspace(0, 1, 7)

            # Extract 'theta' values for the current 'time' from df_result
            # theta_values = df_result[df_result['time'] == time_value][
            #     ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']].values.flatten()

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
        np.savez(f"{output_dir}/observed_{name}_{phantom}.npz", x=observing['position'], t=observing['time'],
                 t_0=observing['t_0'], t_1=observing['t_1'], t_bolus=observing['t_bolus'])

        # Create a figure and three subplots arranged in a row
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot observing['t_inf'] vs observing['time'] on the left subplot
        axs[0].plot(observing['time'], observing['t_0'], 'r-', label='t_inf')
        axs[0].set_xlabel('time')
        axs[0].set_title(r'$y_1(\tau)$')
        # axs[0].set_title('t_0 vs Time')

        # Plot observing['t_sup'] vs observing['time'] on the center subplot
        axs[1].plot(observing['time'], observing['t_1'], 'g-', label='t_sup')
        axs[1].set_xlabel('time')
        # axs[1].set_ylabel('t_1')
        axs[1].set_title(r'$y_2(\tau)$')

        # Plot observing['t_bolus'] vs observing['time'] on the right subplot
        axs[2].plot(observing['time'], observing['t_bolus'], 'b-', label='t_bolus')
        axs[2].set_xlabel('time')
        # axs[2].set_ylabel(r'$t_bolus$')
        axs[2].set_title(r'$y_3(\tau)$')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'{output_dir}/check_{phantom}.png')

        # Show the plot
        plt.show()
