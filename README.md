# PINNs Bioheat Experiments Repository

## Overview

This repository contains code, models, and results for the observation of bioheat transfer problems using Physics-Informed Neural Networks (PINNs) and comparing their performance with ground truth simulations. The project is organized into three main folders: `models`, `tests`, and `src`.

## Folder Structure

### `models/`
This folder contains trained PINNs models, each uniquely identified by a random hash. For each model, the following files are provided:
- A trained model file.
- A plot showing the evolution of the loss components during training.
- A corresponding `.yaml` configuration file that defines the experiment setup.
The unique hash is generated from the values in config.model_properties, so these values are the ones that characterize each model.

### `tests/`
This folder contains the results of the experiments, organized into three categories:
1. **cooling_simulation/**:  
   Contains both MATLAB ground truth and PINNs predictions:
   - `1obs`: PINNs validation with a single observer (comparison with MATLAB ground truth).
   - `8obs`: PINNs with a multiple-model observer, designed to outperform MATLAB.
   
2. **cooling_meas_1/** and **cooling_meas_2/**:  
   These folders contain the PINNs predictions on measurements performed at AMC Hospital.

3. **inverse_problem_cooling_meas_1/** and **inverse_problem_cooling_meas_2/**:  
   Results of inverse problem ran on measurements dataset to retrieve the perfusion value.

### `src/`
Contains the source code for running experiments. Main scripts and folders:
- **`run.py`**: The main script to execute the complete set of experiments.
- **`main.py`**: Script for running individual experiments. Before running `main.py`, modify the experiment parameters in the `config.yaml` file located in the `src` folder.
- **`configs/`**: Contains predefined `.yaml` configuration files for specific experiments. Suffix "enhanced" stands for simulations with increased lambda and upsilon ( the two parameters of the ODE whose too much high value cause MATLAB to stop).
- **`matlab/`**: Contains MATLAB scripts used to generate ground truth simulations.
 Other utility files:
- **`coeff_calc.py`**: The script to load properties from `config.yaml` and calculate the coefficients of the PBHE.
- **`common.py`**: The script to perform some common functions, such as create output folders.
- **`ground_truth.py`**: The script to run matlab ground truth.
- **`import_data.py`**: The script to import measurements.
- **`measurements.py`**: The script to run prediction on measurements.
- **`plots.py`**: The script for plots.
- **`simulation.py`**: The script to run prediction on simulations.
- **`data/`**: Contains measurement data used in the experiments.
- **`heating/`**: Contains scripts about the antenna characterization.
- **`optimization/`**: Contains scripts for optimizing neural networks.
- **`pre-processing/`**: Contains scripts for measurements import.

## Additional Files
Dockerfile is currently not updated.
### `requirements.txt`
This file lists all the Python dependencies required to run the experiments. You can install them by running the following command:
```bash
pip install -r requirements.txt
