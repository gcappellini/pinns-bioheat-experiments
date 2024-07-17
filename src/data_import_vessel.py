import os
import datetime
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
measurements_dir = os.path.join(src_dir, "measurements")

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
    
def find_min_max(timeseries_data):
    min_max_data = {}
    for point, measurements in timeseries_data.items():
        temperatures = [temp for time, temp in measurements]
        min_temp = min(temperatures)
        max_temp = max(temperatures)
        min_max_data[point] = {'min': min_temp, 'max': max_temp}
    return min_max_data


def extract_entries(timeseries_data):
    keys_to_extract = {10: 'y1', 45: 'gt1', 66: 'gt2', 24: 'y2', 31: 'y3'}
    extracted_data = {new_key: timeseries_data.get(old_key, []) for old_key, new_key in keys_to_extract.items()}
    return extracted_data


def create_dataframe(extracted_data):
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
    
    return df

def scale_df(df):
    new_df = pd.DataFrame({'tau': (df["t"]/np.max(df["t"])).round(5)})

    min_temp = np.min(df[['y1', 'gt1', 'gt2', 'y2', 'y3']].min())
    max_temp = np.max(df[['y1', 'gt1', 'gt2', 'y2', 'y3']].max())

    for ei in ['y1', 'gt1', 'gt2', 'y2', 'y3']:
        new_df[ei] = (df[ei]-min_temp)/(max_temp - min_temp)    
    return new_df


file_path = f"{measurements_dir}/vessel/20240522_1.txt"  # Replace with your file path
timeseries_data = load_measurements(file_path)

pickle_file_path = f"{measurements_dir}/vessel/vessel_meas.pkl"
save_to_pickle(timeseries_data, pickle_file_path)


# # Find the minimum and maximum temperatures for each measuring point
# min_max_data = find_min_max(timeseries_data)

# Print the minimum and maximum temperatures for each measuring point
# for point, min_max in min_max_data.items():
#     print(f"Measuring Point {point}: Min Temp = {min_max['min']}, Max Temp = {min_max['max']}")

extracted_data = extract_entries(timeseries_data)
scaled_data = scale_df(extracted_data)

