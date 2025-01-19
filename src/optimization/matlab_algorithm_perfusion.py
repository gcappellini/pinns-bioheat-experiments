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
    uu.check_and_wandb_upload(
        label=label,
        mm_obs_gt=mm_obs_gt,
        system_gt=system_gt,
        conf=config,
        output_dir=output_dir_gt,
        observers_gt=observers_gt
    )
    system_meas, _ = uu.import_testdata(config)
    uu.check_measurements(system_meas, system_gt, output_dir_gt, config)
    extracted = uu.extract_matching([system_gt, system_meas])
    metric = uu.compute_metrics(extracted, [system_gt, system_meas], config, out_dir)
    score = metric["system_meas_L2RE"]
    
    return score


def main():
    """
    Main function to run the testing of the network, MATLAB ground truth, observer checks, and PINNs.
    """
    config = compose(config_name="config_run")
    out_dir = config.output_dir

    perfusions = np.linspace(1e-04, 1e-07, num=10).round(6)
    iterations = []
    scores = []
    for W in perfusions:
        iteration = int(np.argmin(np.abs(perfusions-W)))
        iterations.append(iteration)
        print(f"iteration {iteration}")
        print(f"perfusion {W}")
        config.model_parameters.W_sys = float(W)
        # OmegaConf.save(config, f"{conf_dir}/config_run.yaml")
        score = run_ground_truth(iteration, config, out_dir)
        print(f"score: {score}")
        print("------------------------------------------------")
        scores.append(score)
    
    result = np.hstack((np.array(iterations).reshape(-1, 1), np.array(perfusions).reshape(-1, 1), np.array(scores).reshape(-1, 1)))
    result = result[result[:, 2].argsort()]
    with open(f"{out_dir}/result.txt", "w") as file:
        file.write("Iteration\tPerfusion\tScore\n")
        np.savetxt(file, result, delimiter="\t")





if __name__ == "__main__":
    main()