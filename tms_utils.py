import pandas as pd
import os
import numpy as np
# import deepxde as dde
# import torch

current_file = os.path.abspath(__file__)
script_directory = os.path.dirname(current_file)
meas_dir = f"{script_directory}/measurements"

df = pd.read_csv(f"{meas_dir}/20240403.txt")
df.columns = ['serialNumber', 'sensorDistance', 'sessionDateTime', 'temperatureTimeSeries']
df['sessionDateTime'] = pd.to_datetime(df['sessionDateTime'])

serial_mapping = {
    1000: 'COOLWATER',
    2516: 'T1',
    2515: 'T2',
    1198: 'TA',
    1204: 'TB',
    1200: 'TC',
    1196: 'TD',
    1197: 'TE',
    1205: 'TF',
    1186: 'TG',
    1201: 'TH',
    728: 'TI',
    731: 'TJ',
    720: 'TL',
    723: 'TM',
    725: 'TN',
    701: 'TO',
    729: 'TP',
    732: 'TQ',
    1207: 'TR',
    2508: 'TS',
    2509: 'TT',
    2510: 'TU',
    2511: 'TV',
    2512: 'TW',
    2513: 'TX',
    '1206_1': 'TY',
    '1206_2': 'TZ'
}

# df['serialNumber'] = df['serialNumber'].astype(str)
# # Replace serial numbers with corresponding names
# df['serialNumber'] = df['serialNumber'].map(serial_mapping)

time_temp_pairs = df['temperatureTimeSeries'].str.split('|')

# Initialize an empty dictionary
data_dict = {}

# Loop through each row and extract time-temperature pairs
for index, row in df.iterrows():
    time_temp_data = []
    for pair in time_temp_pairs[index]:
        try:
            time, temp = pair.split()
            time_temp_data.append([int(time), float(temp)])
        except ValueError:
            # print(f"Issue with pair: {pair}. Skipping...")
            continue
    
    # Create a dictionary entry with 'serialNumber', 'sensorDistance', and 'sessionDateTime' as keys
    # and the list of time-temperature pairs as the value
    data_dict[index] = {
        'serialNumber': row['serialNumber'],
        'sensorDistance': row['sensorDistance'],
        'sessionDateTime': row['sessionDateTime'],
        'time_temp_pairs': time_temp_data
    }


