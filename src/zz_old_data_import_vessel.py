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
    keys_to_extract = {10: 'y1', 45: 'gt1', 66: 'gt2', 24: 'y2', 31: 'y3'}
    extracted_data = {new_key: timeseries_data.get(old_key, []) for old_key, new_key in keys_to_extract.items()}

    # Create a list of all unique times
    all_times = sorted(set(time for times in extracted_data.values() for time, temp in times))

    # return all_times
    
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

    threshold = 1 if el==3 else 100

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


file_path = f"{src_dir}/data/measurements/vessel/20240522_1.txt"  # Replace with your file path
timeseries_data = load_measurements(file_path)

times = [
    (0, 30 * 60),
    (30 * 60, (30+27) * 60),
    ((30+27) * 60, (30+27+29) * 60),
    ((30+27+29) * 60, (30+27+29+47) * 60)
]

for el in range(len(times)):
    tmin, tmax = times[el]
    df = extract_entries(timeseries_data, tmin, tmax)

    scaled_data = scale_df(df)

    pickle_file_path = f"{src_dir}/data/measurements/vessel/{el}.pkl"
    save_to_pickle(scaled_data, pickle_file_path)


    fig, ax = plt.subplots()

    ax.plot(scaled_data['tau'], scaled_data['y1'], label=f'y1', marker='x')# alpha=1.0, linewidth=.7)
    ax.plot(scaled_data['tau'], scaled_data['gt1'], label=f'gt1', alpha=1.0, linewidth=.7)
    ax.plot(scaled_data['tau'], scaled_data['gt2'], label=f'gt2', alpha=1.0, linewidth=.7)
    ax.plot(scaled_data['tau'], scaled_data['y2'], label=f'y2', alpha=1.0, linewidth=.7)
    ax.plot(scaled_data['tau'], scaled_data['y3'], label=f'y3', alpha=1.0, linewidth=.7)
    ax.legend()
    ax.set_title(f"Test {el}")

    # ax.plot(range(len(df['t'])), df['t'], alpha=1.0, linewidth=.7)
    # plt.show()
    plt.savefig(f"{src_dir}/data/measurements/vessel/test_{el}.png", dpi=1200)
    plt.close()

    

