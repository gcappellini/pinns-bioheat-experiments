import numpy as np
import utils as uu
import os
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)


exps = ["cooling_meas_2", "heating_meas_1","heating_meas_2","heating_meas_3"]
for ciccio in exps:
    a = uu.load_from_pickle(f"{src_dir}/data/vessel/{ciccio}.pkl")
    # Assuming your dataframe is named df
    a.to_csv(f'{src_dir}/data/vessel/{ciccio}.txt', sep='\t', index=False, header=False)
