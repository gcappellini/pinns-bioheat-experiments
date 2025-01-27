import os
import numpy as np
from omegaconf import OmegaConf
from hydra import compose
import utils as uu
import common as co
import plots as pp

# Directories Setup
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
conf_dir = os.path.join(src_dir, "configs")
os.makedirs(tests_dir, exist_ok=True)


def run_ground_truth(iter, config, out_dir):
    """Run MATLAB ground truth simulation, load data, and plot results."""
    label = f"ground_truth_{iter}"
    output_dir_gt, config_matlab = co.set_run(out_dir, config, label)
    uu.run_matlab_ground_truth()
    system_gt, observers_gt, mm_obs_gt = uu.gen_testdata(config_matlab, path=output_dir_gt)
    system_meas, _ = uu.import_testdata(config)
    
    # uu.check_measurements(system_meas, system_gt, output_dir_gt, config)

    points = uu.get_tc_positions()
    x_gt2 = points[1]
    x_gt1 = points[2]

    mask_all = np.isin(system_gt["grid"][:, 0], [x_gt2, x_gt1, 0.0])
    system_all = {"grid": system_gt["grid"][mask_all], "theta": system_gt["theta"][mask_all]}
    systems_gt, mm_obs_gt = uu.calculate_l2(system_meas, [system_all], mm_obs_gt)
    system_gt=systems_gt[0]

    mask_gt2 = np.isin(system_gt["grid"][:, 0], [x_gt2, 0.0])
    system_gt2 = {"grid": system_gt["grid"][mask_gt2], "theta": system_gt["theta"][mask_gt2]}
    systems_gt2, mm_obs_gt = uu.calculate_l2(system_meas, [system_gt2], mm_obs_gt)
    system_gt2=systems_gt2[0]

    mask_gt1 = np.isin(system_gt["grid"][:, 0], [x_gt1, 0.0])
    system_gt1 = {"grid": system_gt["grid"][mask_gt1], "theta": system_gt["theta"][mask_gt1]}
    systems_gt1, mm_obs_gt = uu.calculate_l2(system_meas, [system_gt1], mm_obs_gt)
    system_gt1=systems_gt1[0]

    score_all = np.sum(system_gt["L2_err"])
    score_gt1 = np.sum(system_gt1["L2_err"])
    score_gt2 = np.sum(system_gt2["L2_err"])
    
    return [score_all, score_gt1, score_gt2]


def main():
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    exps = ["meas_cool_1", "meas_cool_2"]

    for exp in exps:
        conf = OmegaConf.load(f"{conf_dir}/config_run.yaml")
        conf.experiment.run = exp
        OmegaConf.save(conf, f"{conf_dir}/config_run.yaml")
        config = compose(config_name="config_run")
        props = OmegaConf.to_container(config.model_properties, resolve=True)
        pars = OmegaConf.to_container(config.model_parameters, resolve=True)

        out_dir = os.path.abspath(config.run.dir)
        os.makedirs(out_dir, exist_ok=True)

        perfusions = np.logspace(np.log10(1e-06), np.log10(3e-04), num=50).round(9)

        # values = ["iteration", "perfusion", "score_all", "score_gt1", "score_gt2"]
        values = []
        for i, W in enumerate(perfusions):

            print(f"iteration {i}: W={W}")
            config.model_parameters.W_sys = float(W)
            # OmegaConf.save(config, f"{conf_dir}/config_run.yaml")

            points = run_ground_truth(i, config, out_dir)

            # values.append(np.array([i, W, *points]))
            values.append([i, W, *points])
        
        with open(f"{out_dir}/result_{config.experiment.run}.txt", "w") as file:
            # file.write("Iteration\tPerfusion\tScore\n")
            np.savetxt(file, values, delimiter="\t")

        values = np.array(values)
        iters = values[:, 0].astype(int)
        perfusions = values[:, 1]
        scores_all = values[:, 2]
        scores_gt1 = values[:, 3]
        scores_gt2 = values[:, 4]

        pp.plot_generic(
            x=[perfusions, perfusions, perfusions],   # Provide time values for each line (either one for each model or just one for single prediction)
            y=[scores_all, scores_gt1, scores_gt2],       # Multiple L2 error lines to plot
            title="Prediction error norm",
            xlabel="Perfusion",
            ylabel=r"$L^2$ norm",
            legend_labels=["all", "gt1", "gt2"],  # Labels for the legend
            size=(6, 5),
            filename=f"{out_dir}/matlab_alg_{config.experiment.run}_zoom.png",
            colors=["C0", "C1", "C2"],
            log_xscale=True
        )



if __name__ == "__main__":
    main()