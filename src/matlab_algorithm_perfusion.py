import os
import numpy as np
from omegaconf import OmegaConf
from hydra import compose
import utils as uu
import common as co

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
    uu.check_measurements(system_meas, system_gt, output_dir_gt, config)

    metric = uu.compute_metrics([system_gt, system_meas], config, out_dir)
    print(metric.keys())
    score_all = metric["system_meas_L2RE"]
    points = uu.get_tc_positions()
    x_gt2 = points[1]
    x_gt1 = points[2]

    mask_gt2 = [system_meas["grid"][:, 0] != x_gt2]
    mask_gt1 = [system_meas["grid"][:, 0] != x_gt1]

    system_meas_gt1 = {"grid": system_meas["grid"][mask_gt1], "theta": system_meas["theta"][mask_gt1]}
    system_meas_gt2 = {"grid": system_meas["grid"][mask_gt2], "theta": system_meas["theta"][mask_gt2]}
    
    metric_gt1 = uu.compute_metrics([system_gt, system_meas_gt1], config, out_dir)
    metric_gt2 = uu.compute_metrics([system_gt, system_meas_gt2], config, out_dir)
    
    score_gt1 = metric_gt1["system_meas_L2RE"]
    score_gt2 = metric_gt2["system_meas_L2RE"]
    
    return score_all, score_gt1, score_gt2


def main():
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    config = compose(config_name="config_run")
    props = OmegaConf.to_container(config.model_properties, resolve=True)
    pars = OmegaConf.to_container(config.model_parameters, resolve=True)

    out_dir = os.path.abspath(config.run.dir)
    os.makedirs(out_dir, exist_ok=True)

    perfusions = np.linspace(pars["W_min"], pars["W_max"], num=3).round(6)

    values = []
    for i, W in enumerate(perfusions):

        print(f"iteration {i}")
        print(f"perfusion {W}")
        config.model_parameters.W_sys = float(W)
        # OmegaConf.save(config, f"{conf_dir}/config_run.yaml")

        points = run_ground_truth(i, config, out_dir)

        # values.append([i, W, *points])
    
    # result = np.hstack((np.array(iterations).reshape(-1, 1), np.array(perfusions).reshape(-1, 1), np.array(scores).reshape(-1, 1)))
    # result = result[result[:, 2].argsort()]
    
    with open(f"{out_dir}/result.txt", "w") as file:
        file.write("Iteration\tPerfusion\tScore\n")
        np.savetxt(file, values, delimiter="\t")





if __name__ == "__main__":
    main()