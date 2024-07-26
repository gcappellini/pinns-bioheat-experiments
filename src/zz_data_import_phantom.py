import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
measurements_dir = os.path.join(src_dir, "measurements")

def parse_temperature_file(file_path):
    data = []
    
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split(', ')
            serial_number = parts[0]
            sensor_distance = float(parts[1])
            session_datetime = datetime.strptime(parts[2], '%Y-%m-%d %H:%M:%S.%f')
            time_series_data = parts[3]
            
            # Parse the time series data
            time_series_list = time_series_data.split('|')
            for entry in time_series_list:
                if entry.strip():
                    time, temp = entry.split()
                    datetime_entry = session_datetime + timedelta(seconds=int(time))
                    data.append([serial_number, sensor_distance, datetime_entry, float(temp)])
    
    # # Create a DataFrame
    df = pd.DataFrame(data, columns=['SerialNumber', 'SensorDistance', 'DateTime', 'Temperature'])
    return df
    # return parts


# Example usage
file_path = f"{measurements_dir}/phantom/20240403.txt"
df = parse_temperature_file(file_path)
df.to_pickle(f"{measurements_dir}/phantom/phantom_meas.pkl")

# next: map the phantom - a dictionary (catheter, point) = time series measurements:
# 1) 



# next (for 3D): transformation (catheter, point) -> (x, y, z)
