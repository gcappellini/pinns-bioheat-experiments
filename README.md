# PINNs Bioheat Experiments Repository

## Overview

This repository contains code, models, and results for the observation of bioheat transfer problems using Physics-Informed Neural Networks (PINNs) and comparing their performance with ground truth simulations. The project is organized into three main folders: `models`, `tests`, and `src`.

## Folder Structure

### `models/`
This folder contains trained PINNs models, each uniquely identified by a random hash. For each model, the following files are provided:
- A trained model file.
- A plot showing the evolution of the loss components during training.
- A corresponding `.yaml` configuration file that defines the experiment setup.

### `tests/`
This folder contains the results of the experiments, organized into three categories:
1. **cooling_simulation/**:  
   Contains both MATLAB ground truth and PINNs predictions:
   - `1obs`: PINNs validation with a single observer (comparison with MATLAB ground truth).
   - `8obs`: PINNs with a multiple-model observer, designed to outperform MATLAB.
   
2. **cooling_meas_1/** and **cooling_meas_2/**:  
   These folders contain the PINNs predictions on measurements performed at AMC Hospital.

### `src/`
Contains the source code for running experiments, along with additional utility files:
- **`run.py`**: The main script to execute the complete set of experiments.
- **`main.py`**: Script for running individual experiments. Before running `main.py`, modify the experiment parameters in the `config.yaml` file located in the `src` folder.
- **`configs/`**: Contains predefined `.yaml` configuration files for specific experiments.
- **`matlab/`**: Contains MATLAB scripts used to generate ground truth simulations.
- **`data/`**: Contains measurement data used in the experiments.
- Other subfolders are not currently relevant.

## Additional Files

### `requirements.txt`
This file lists all the Python dependencies required to run the experiments. You can install them by running the following command:
```bash
pip install -r requirements.txt