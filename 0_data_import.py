import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)

nn = ["240124", "240125", "240125b", "240130", "240202"]
pp = ["P1", "P2"]

output_dir = f"{script_directory}/measurements"

max_T = 37.8
min_T = 22.0

os.makedirs(output_dir, exist_ok=True)


for date in nn:
    for phantom in pp:

        # Rename according to provided mapping
        if date == "240125b" and phantom == "P1":
            name = "AX1"
        elif date == "240124" and phantom == "P1":
            name = "AX2"
        elif date == "240125b" and phantom == "P2":
            name = "BX1"
        elif date == "240125" and phantom == "P2":
            name = "BX2"
        elif date == "240130" and phantom == "P1":
            name = "AY1"
        elif date == "240202" and phantom == "P1":
            name = "AY2"
        elif date == "240130" and phantom == "P2":
            name = "BY1"
        elif date == "240124" and phantom == "P2":
            name = "BY2"
        else:
            # Skip storing 240202_P2 and 240125_P1
            continue

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

labels = [["AX1", "AX2", "BX1", "BX2"], ["AY1", "AY2", "BY1", "BY2"]]

# Create figure 1
fig1, axs1 = plt.subplots(2, 2, figsize=(13, 7))

# Load and plot data for figure 1
for j, label in enumerate(labels[0]):
    # Load the saved data
    data = np.load(f"{output_dir}/meas_{label}.npz")
    x = data['x']
    t = data['t']
    theta = data['theta']

    la = len(np.unique(x))
    le = len(np.unique(t))

    # Plot theta vs x and t using imshow
    axs1[j//2, j%2].scatter(x, t, c=theta, cmap='inferno', vmin=0, vmax=1, s=60, marker='o', edgecolors='none')
    axs1[j//2, j%2].set_title(f"{label}", fontweight="bold")
    axs1[j//2, j%2].set_xlabel('Z', fontsize=12)
    axs1[j//2, j%2].set_ylabel(r'$\tau$', fontsize=12)
    plt.colorbar(axs1[j//2, j%2].scatter([], [], c=[], cmap='inferno', vmin=0, vmax=1), ax=axs1[j//2, j%2], label=r'$\theta$')


# Adjust layout and save figure 1
plt.tight_layout()
plt.savefig(f'{output_dir}/meas_X.png')
plt.close()

# Create figure 2
fig2, axs2 = plt.subplots(2, 2, figsize=(13, 7))

# Load and plot data for figure 2
for j, label in enumerate(labels[1]):
    # Load the saved data
    data = np.load(f"{output_dir}/meas_{label}.npz")
    x = data['x']
    t = data['t']
    theta = data['theta']

    la = len(np.unique(x))
    le = len(np.unique(t))

    d = theta.max() if theta.max() > 1 else 1
    # Plot theta vs x and t using imshow
    axs2[j//2, j%2].scatter(x, t, c=theta, cmap='inferno', vmin=0, vmax=d, s=60, marker='o', edgecolors='none')
    axs2[j//2, j%2].set_title(f"{label}", fontweight="bold")
    axs2[j//2, j%2].set_xlabel('Z', fontsize=12)
    axs2[j//2, j%2].set_ylabel(r'$\tau$', fontsize=12)
    plt.colorbar(axs2[j//2, j%2].scatter([], [], c=[], cmap='inferno', vmin=0, vmax=d), ax=axs2[j//2, j%2], label=r'$\theta$')

# Adjust layout and save figure 2
plt.tight_layout()
plt.savefig(f'{output_dir}/meas_Y.png')
plt.close()

# Create figure 3
fig3, axs3 = plt.subplots(2, 2, figsize=(13, 7))

# Load and plot data for figure 3
# Load and plot data for figure 3
for i, label in enumerate(labels[0]):
    # Load the saved data
    obs = np.load(f"{output_dir}/obs_{label}.npz")
    t, t_0, t_1, t_bolus = obs["t"], obs["t_0"], obs["t_1"], obs["t_bolus"] 

    # Plot t_0, t_1, and t_bolus against t on each subplot
    axs3[i//2, i%2].plot(t, t_0, 'r-', label=r'$\tilde y_1(\tau)$')
    axs3[i//2, i%2].plot(t, t_1, 'g-', label=r'$\tilde y_2(\tau)$')
    axs3[i//2, i%2].plot(t, t_bolus, 'b-', label=r'$\tilde y_3(\tau)$')
    axs3[i//2, i%2].set_xlabel(r"$\tau$", fontsize=12)
    axs3[i//2, i%2].set_ylabel(r"$\tilde y(\tau)$", fontsize=12)
    axs3[i//2, i%2].set_title(f"{label}", fontsize=14, fontweight="bold")
    axs3[i//2, i%2].tick_params(axis='both', which='major', labelsize=10)
    axs3[i//2, i%2].legend()

# Adjust layout
plt.tight_layout()
plt.savefig(f'{output_dir}/obs_X.png')
plt.show()
plt.close()


# Create figure 3
fig4, axs4 = plt.subplots(2, 2, figsize=(13, 7))

# Load and plot data for figure 4
for i, label in enumerate(labels[1]):
    # Load the saved data
    obs = np.load(f"{output_dir}/obs_{label}.npz")
    t, t_0, t_1, t_bolus = obs["t"], obs["t_0"], obs["t_1"], obs["t_bolus"] 

    # Plot observing['t_inf'] vs observing['time'] on the left subplot
    axs4[i//2, i%2].plot(t, t_0, 'r-', label=r'$\tilde y_1(\tau)$')
    axs4[i//2, i%2].plot(t, t_1, 'g-', label=r'$\tilde y_2(\tau)$')
    axs4[i//2, i%2].plot(t, t_bolus, 'b-', label=r'$\tilde y_3(\tau)$')
    axs4[i//2, i%2].set_xlabel(r"$\tau$", fontsize=12)
    axs4[i//2, i%2].set_ylabel(r"$\tilde y(\tau)$", fontsize=12)
    axs4[i//2, i%2].set_title(f"{label}", fontsize=14, fontweight="bold")
    axs4[i//2, i%2].tick_params(axis='both', which='major', labelsize=10)
    axs4[i//2, i%2].legend()

# Adjust layout
plt.tight_layout()
plt.savefig(f'{output_dir}/obs_Y.png')
plt.show()
plt.close()


