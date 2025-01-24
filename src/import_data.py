import os
import numpy as np
import utils as uu
from omegaconf import OmegaConf
import plots as pp

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

def main(meas, rescale, save_pickle, show_y3=False, threshold=0.0):

    date = meas.date
    start_min = meas.start_min
    end_min = meas.end_min
    string = meas.string
    name = meas.title

    file_path = f"{src_dir}/data/vessel/{date}.txt"

    timeseries_data = uu.load_measurements(file_path)
    df = uu.extract_entries(timeseries_data, start_min*60, end_min*60, threshold)



    df1 = uu.scale_df(df)

    if save_pickle:
        uu.save_to_pickle(df1, f"{src_dir}/data/vessel/{string}.pkl")
        df1.to_csv(f"{src_dir}/data/vessel/{string}.txt", header=False, index=False, sep="\t")

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
    
    return df['y1'][0],df['gt1'][0],df['gt2'][0], df['y2'][0], df['y3'][0]



if __name__ == "__main__":
    conf = OmegaConf.load(f"{src_dir}/configs/config_run.yaml")
    rescale = conf.plot.rescale
    save_pickle = conf.experiment_type.save_pickle

    # Loop through all experiment types except antenna_characterization
    for experiment_type in conf.experiment_type.keys():
        
        if experiment_type == 'antenna_characterization':
            antenna_exp = conf.experiment_type.antenna_characterization
            threshold = antenna_exp['threshold']
            show_y3 = antenna_exp['show_y3']
            main(antenna_exp, rescale, save_pickle, show_y3, threshold)  # Use the extracted values
            continue

        if experiment_type.startswith('meas_'):
            meas = getattr(conf.experiment_type, experiment_type)
            Ty10, Tgt10, Tgt20, Ty20, Ty30 = main(meas, rescale, save_pickle)
            meas.Ty10 = float(round(Ty10, 2))
            meas.Tgt10 = float(round(Tgt10, 2))
            meas.Tgt20 = float(round(Tgt20, 2))
            meas.Ty20 = float(round(Ty20, 2))
            meas.Ty30 = float(round(Ty30, 2))

            OmegaConf.update(conf, f"experiment_type.{experiment_type}", meas)
            OmegaConf.save(conf, f"{src_dir}/configs/config_run.yaml")







    

