import os
import numpy as np
import utils as uu
from omegaconf import OmegaConf
import plots as pp

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

def main(meas, rescale, save_pickle, show_y3, threshold):

    date = meas.date
    start_min = meas.start_min
    end_min = meas.end_min
    string = meas.string
    name = meas.title

    file_path = f"{src_dir}/data/vessel/{date}.txt"

    timeseries_data = uu.load_measurements(file_path)
    df = uu.extract_entries(timeseries_data, start_min*60, end_min*60, threshold)

    # print(df['y1'][0],df['gt1'][0],df['gt2'][0], df['y2'][0], df['y3'][0])

    df1 = uu.scale_df(df)

    if save_pickle:
        uu.save_to_pickle(df1, f"{src_dir}/data/vessel/{string}.pkl")


    title=f"{name}"
    xlabel="Time (min)"

    _, _, ylabel = uu.get_scaled_labels(rescale)

    if show_y3:
        x_vals = np.vstack(([df['t']/60]*5)).T
        y_vals = np.vstack((df['y1'],df['gt1'],df['gt2'],df['y2'],df['y3'])).T
        labels = ["y1", "gt1", "gt2", "y2", "y3"]
        colors = conf.plot.colors.measuring_points
        linestyles = conf.plot.linestyles.measuring_points

    else:
        x_vals = np.vstack(([df['t']/60]*4)).T
        y_vals = np.vstack((df['y1'],df['gt1'],df['gt2'],df['y2'])).T
        labels = ["y1", "gt1", "gt2", "y2"]
        colors = conf.plot.colors.measuring_points[:-1]
        linestyles = conf.plot.linestyles.measuring_points[:-1]


    pp.plot_generic(x_vals.T, 
                    y_vals.T, 
                    title, 
                    xlabel, 
                    ylabel, 
                    labels, 
                    size = (10, 5),
                    filename=f"{src_dir}/data/vessel/{string}.png", 
                    colors=colors, linestyles = linestyles)



if __name__ == "__main__":
    conf = OmegaConf.load(f"{src_dir}/config.yaml")
    rescale = conf.plot.rescale
    save_pickle = conf.experiment.save_pickle

    # Loop through all experiment types except antenna_characterization
    for experiment_type in conf.experiment.type.keys():
        
        if experiment_type == 'antenna_characterization':
            antenna_exp = conf.experiment.type.antenna_characterization
            threshold = antenna_exp['threshold']
            show_y3 = antenna_exp['show_y3']
            main(antenna_exp, rescale, save_pickle, show_y3, threshold)  # Use the extracted values
            continue

        # For other experiment types
        experiment_group = getattr(conf.experiment.type, experiment_type)
        threshold = experiment_group['threshold']
        show_y3 = experiment_group['show_y3']

        # Process each measurement in the experiment group
        for meas_key in experiment_group.keys():
            if meas_key.startswith('meas_'):
                meas = getattr(experiment_group, meas_key)
                main(meas, rescale, save_pickle, show_y3, threshold)







    

