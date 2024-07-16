import os
from datetime import datetime
import pickle

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
measurements_dir = os.path.join(src_dir, "measurements")

def parse_line(line):
    parts = line.strip().split(', ')
    time_str = parts[0].split()[1]  # Extract the time (HH:MM:SS.microsecond)
    time = datetime.strptime(time_str, '%H:%M:%S.%f').time()
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


# Example usage
file_path = f"{measurements_dir}/vessel/20240522_1.txt"  # Replace with your file path
timeseries_data = load_measurements(file_path)

pickle_file_path = f"{measurements_dir}/vessel/vessel_meas.pkl"
save_to_pickle(timeseries_data, pickle_file_path)


# Find the minimum and maximum temperatures for each measuring point
min_max_data = find_min_max(timeseries_data)

# Print the minimum and maximum temperatures for each measuring point
for point, min_max in min_max_data.items():
    print(f"Measuring Point {point}: Min Temp = {min_max['min']}, Max Temp = {min_max['max']}")
# # Print the timeseries for each point
# for point, measurements in timeseries_data.items():
#     print(f"Measuring Point {point}:")
#     for time, temp in measurements:
#         print(f"  {time}: {temp}")



# next (for 3D): transformation (catheter, point) -> (x, y, z)
