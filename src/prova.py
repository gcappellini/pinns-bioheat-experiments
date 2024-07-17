import data_import_vessel as utils
import os
import pandas as pd
import datetime
import numpy as np

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
measurements_dir = os.path.join(src_dir, "measurements")


file_path = f"{measurements_dir}/vessel/20240522_1.txt"  # Replace with your file path
timeseries_data = utils.load_measurements(file_path)

extracted_data = utils.extract_entries(timeseries_data)

# Create a DataFrame from the extracted data
df = utils.create_dataframe(extracted_data)

new_df = utils.scale_df(df)

# Print the DataFrame
print(new_df)