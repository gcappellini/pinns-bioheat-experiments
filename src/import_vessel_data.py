import os
import datetime
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

def parse_line(line):
    parts = line.strip().split(', ')
    time_str = parts[0].split()[1]  # Extract the time (HH:MM:SS.microsecond)
    time = datetime.datetime.strptime(time_str, '%H:%M:%S.%f').time()
    measurements = parts[2:]  # Skip date and 'Temperature' keyword
    data = {}
    for i in range(0, len(measurements), 2):
        point = int(measurements[i])
        temperature = float(measurements[i + 1])
        if point not in data:
            data[point] = []
        data[point].append((time, temperature))
    return data

def load_measurements(file_path):
    timeseries_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = parse_line(line)
            for point, measurements in data.items():
                if point not in timeseries_data:
                    timeseries_data[point] = []
                timeseries_data[point].extend(measurements)
    return timeseries_data


def save_to_pickle(data, file_path):
    with open(file_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def load_from_pickle(file_path):
    with open(file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)
    

def extract_entries(timeseries_data, tmin, tmax):
    keys_to_extract = {10: 'y1', 45: 'gt1', 66: 'gt2', 24: 'y2', 31: 'y3', 39:'bol_out'}
    extracted_data = {new_key: timeseries_data.get(old_key, []) for old_key, new_key in keys_to_extract.items()}

    # Create a list of all unique times
    all_times = sorted(set(time for times in extracted_data.values() for time, temp in times))
    
    # # Normalize times to seconds, starting from zero
    start_time = all_times[0]
    all_times_in_seconds = [(datetime.datetime.combine(datetime.date.today(), time) - 
                             datetime.datetime.combine(datetime.date.today(), start_time)).total_seconds() 
                            for time in all_times]
    
    # Initialize the dataframe
    df = pd.DataFrame({'t': np.array(all_times_in_seconds).round()})
    
    # Populate the dataframe with temperatures
    for key, timeseries in extracted_data.items():
        temp_dict = {time: temp for time, temp in timeseries}
        df[key] = [temp_dict.get(time, float('nan')) for time in all_times]
    
    df = df[(df['t']>tmin) & (df['t']<tmax)].reset_index(drop=True)

    # return df
    df['time_diff'] = df['t'].diff()#.dt.total_seconds()

    threshold = np.where((df['t'] > 2 * 60) & (df['t'] < 83 * 60), 50, 1)

    # Identify the indices where a new interval starts
    new_intervals = df[df['time_diff'] > threshold].index

    # Include the first index as the start of the first interval
    new_intervals = [0] + list(new_intervals)

    # Create an empty list to store the last measurements of each interval
    last_measurements = []

    # Extract the last measurement from each interval
    for i in range(len(new_intervals)):
        start_idx = new_intervals[i]
        end_idx = new_intervals[i + 1] - 1 if i + 1 < len(new_intervals) else len(df) - 1
        last_measurements.append(df.iloc[end_idx])

    # Create the df_short DataFrame from the list of last measurements
    df_short = pd.DataFrame(last_measurements).drop(columns=['time_diff']).reset_index(drop=True)

    return df_short



def scale_df(df):
    time = df['t']-df['t'][0]
    new_df = pd.DataFrame({'tau': (time/np.max(time)).round(5)})

    min_temp = np.min(df[['y1', 'gt1', 'gt2', 'y2', 'y3']].min())
    max_temp = np.max(df[['y1', 'gt1', 'gt2', 'y2', 'y3']].max())

    for ei in ['y1', 'gt1', 'gt2', 'y2', 'y3']:
        new_df[ei] = (df[ei]-min_temp)/(max_temp - min_temp)    
    return new_df


file_path = f"{src_dir}/data/vessel/20240522_1.txt"
timeseries_data = load_measurements(file_path)
df = extract_entries(timeseries_data, 83*60, 4*60*60)
print(df['gt2'].max(), df['y1'].min(), df['gt2'].max() - df['y1'].min())
df1 = scale_df(df)
save_to_pickle(df1, f"{src_dir}/cooling_scaled.pkl")
# df1.to_csv(f"{src_dir}/cooling_scaled.txt", index=False, header=False)

fig, ax = plt.subplots(figsize=(12, 6))  # Stretching layout horizontally

# Plotting data with specified attributes
ax.plot(df['t']/60, df['y1'], label='y1', marker='x')
ax.plot(df['t']/60, df['gt1'], label='gt1', alpha=1.0, linewidth=0.7)
ax.plot(df['t']/60, df['gt2'], label='gt2', alpha=1.0, linewidth=0.7)
ax.plot(df['t']/60, df['y2'], label='y2', alpha=1.0, linewidth=0.7)
ax.plot(df['t']/60, df['y3'], label='y3', alpha=1.0, linewidth=0.7)
# ax.plot(df['t']/60, df['bol_out'], label='bolus outlet', alpha=1.0, linewidth=0.7)

# Add vertical dashed red lines with labels on the plot
# ax.axvline(x=2, color='red', linestyle='--', linewidth=1.1)
# ax.text(2.5, 35.8, 'RF on,\nmax perfusion', color='red', fontsize=10, verticalalignment='top')

# ax.axvline(x=29, color='red', linestyle='--', linewidth=1.1)
# ax.text(29.5, 35.8, 'Min perfusion', color='red', fontsize=10, verticalalignment='top')

# ax.axvline(x=56, color='red', linestyle='--', linewidth=1.1)
# ax.text(56.5, 35.8, 'Zero perfusion', color='red', fontsize=10, verticalalignment='top')

# ax.axvline(x=83, color='red', linestyle='--', linewidth=1.1)
# ax.text(83.5, 35.8, 'RF off, max perfusion', color='red', fontsize=10, verticalalignment='top')

# Adding legend for the plotted data (excluding the vertical lines)
ax.legend()

# Setting title and labels with modifications
ax.set_title("Cooling Experiment", fontweight='bold')
ax.set_xlabel("Time (min)", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
# ax.set_xlim(0, 234)

# Adjust layout for better horizontal stretching
plt.tight_layout()

# Display and save plot

# plt.savefig(f"{src_dir}/data/vessel/bolus.png", dpi=120)
plt.show()
plt.close()
plt.clf()

# # Plotting scaled data with specified attributes
# ax.plot(df1['tau'], df1['y1'], label='y1', marker='x')
# ax.plot(df1['tau'], df1['gt1'], label='gt1', alpha=1.0, linewidth=0.7)
# ax.plot(df1['tau'], df1['gt2'], label='gt2', alpha=1.0, linewidth=0.7)
# ax.plot(df1['tau'], df1['y2'], label='y2', alpha=1.0, linewidth=0.7)
# ax.plot(df1['tau'], df1['y3'], label='y3', alpha=1.0, linewidth=0.7)
# # ax.plot(df['t']/60, df['bol_out'], label='bolus outlet', alpha=1.0, linewidth=0.7)


# # Adding legend for the plotted data (excluding the vertical lines)
# ax.legend()

# # Setting title and labels with modifications
# ax.set_title("Cooling Experiment", fontweight='bold')
# ax.set_xlabel(r"$\tau$", fontsize=12)
# ax.set_ylabel(r"$\theta$", fontsize=12)
# # ax.set_xlim(0, 234)

# # Adjust layout for better horizontal stretching
# plt.tight_layout()

# # Display and save plot

# # plt.savefig(f"{src_dir}/data/vessel/bolus.png", dpi=120)
# plt.show()
# # plt.close()
# # plt.clf()


    

