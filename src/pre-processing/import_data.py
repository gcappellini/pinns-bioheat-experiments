import os
import numpy as np
import utils as uu
from omegaconf import OmegaConf
import plots as pp

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

def main(meas, rescale, save_pickle):

    date = meas.date
    start_min = meas.start_min
    end_min = meas.end_min
    string = meas.string
    name = meas.title
    phantom = meas.phantom if hasattr(meas, 'phantom') else None
    if phantom:
        name = f"{name} {phantom}"
        string = f"{string}_{phantom}"

    keys_vessel = {10:'y1', 45:'gt1', 66:'gt', 24:'y2', 31:'y3', 37:'bol_out'}

    keys_agar_A = {
        0: 'A0', 1: 'A1', 2: 'A2', 3: 'A3', 4: 'A4', 5: 'A5', 6: 'A6',
        31: 'bol1_u', 46: 'bol1_d'
    }
    keys_agar_B = {
        14: 'B0', 15: 'B1', 16: 'B2', 17: 'B3', 18: 'B4', 19: 'B5', 20: 'B6',
        59: 'bol2_u', 74: 'bol2_d'
    }

    
    keys_agar = keys_agar_A if phantom == "A" else keys_agar_B
    keys_to_extract = keys_agar if string.startswith("agar") else keys_vessel

    folder = "agar" if string.startswith("agar") else "vessel"
    file_path = f"{src_dir}/data/{folder}"

    timeseries_data = uu.load_measurements(f"{file_path}/{date}.txt")
    df = uu.extract_entries(timeseries_data, start_min*60, end_min*60, keys_to_extract=keys_to_extract)



    # if save_pickle:
        # df1 = uu.scale_df(df)
    #     uu.save_to_pickle(df1, f"{file_path}/{string}.pkl")
    #     df1.to_csv(f"{file_path}/{string}.txt", header=False, index=False, sep="\t")

    title=f"{name}"
    xlabel="Time (min)"

    _, _, ylabel = uu.get_scaled_labels(rescale)

    x_vals = np.vstack(([(df['t']-df['t'][0])/60]*len(keys_to_extract))).T
    labels = list(keys_to_extract.values())
    # labels = ["y1","gt","y2"]
    y_vals = np.vstack([df[label] for label in labels]).T
    
    plot_params = uu.get_plot_params(conf)
    colors = [plot_params[pt_lbl]['color'] for pt_lbl in labels]
    linestyles = ['-']*len(labels)


    pp.plot_generic(x_vals.T, 
                    y_vals.T, 
                    title, 
                    xlabel, 
                    ylabel, 
                    labels, 
                    size = (10, 5),
                    filename=f"{file_path}/{string}.png", 
                    colors=colors, linestyles = linestyles)
    
    # return df['y1'][0],df['gt1'][0],df['gt'][0], df['y2'][0], df['y3'][0]



if __name__ == "__main__":
    conf = OmegaConf.load(f"{src_dir}/configs/config_run.yaml")
    rescale = conf.plot.rescale
    save_pickle = conf.experiment_type.save_pickle

    # Loop through all experiment types except antenna_characterization
    # for experiment_type in conf.experiment_type.keys():
    for i in range(1, 6):
        experiment_type = f"agar_{i}"
        
        if experiment_type == 'antenna_characterization':
            antenna_exp = conf.experiment_type.antenna_characterization
            threshold = antenna_exp['threshold']
            show_y3 = antenna_exp['show_y3']
            main(antenna_exp, rescale, save_pickle, show_y3, threshold)  # Use the extracted values
            continue

        elif experiment_type.startswith("meas_"):
            meas = getattr(conf.experiment_type, experiment_type)
            Ty10, Tgt10, Tgt20, Ty20, Ty30 = main(meas, rescale, save_pickle)
            meas.Ty10 = float(round(Ty10, 2))
            meas.Tgt10 = float(round(Tgt10, 2))
            meas.Tgt20 = float(round(Tgt20, 2))
            meas.Ty20 = float(round(Ty20, 2))
            meas.Ty30 = float(21.5)

            OmegaConf.update(conf, f"experiment_type.{experiment_type}", meas)
            OmegaConf.save(conf, f"{src_dir}/configs/config_run.yaml")
        
        else:
            meas = getattr(conf.experiment_type, experiment_type)
            meas.phantom = "B"
            main(meas, rescale, save_pickle)
            OmegaConf.update(conf, f"experiment_type.{experiment_type}", meas)
            OmegaConf.save(conf, f"{src_dir}/configs/config_run.yaml")
        








    

